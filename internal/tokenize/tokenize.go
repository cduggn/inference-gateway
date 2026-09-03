// Package tokenize turns chat messages into token ids.
//
// LIBRARY GAP. Python has transformers.AutoTokenizer, which loads the exact
// tokenizer the served model was trained with. Go has no equivalent of
// comparable quality. The options, none of them drop-in:
//
//   - github.com/daulet/tokenizers  - cgo bindings to HuggingFace's Rust
//     tokenizers. Faithful, but drags cgo into the build.
//   - github.com/sugarme/tokenizer  - pure Go, partial coverage, lags upstream.
//   - github.com/tiktoken-go/tokenizer - correct, but OpenAI BPE only.
//
// The gateway only needs token *counts* (for quota and capacity math) and
// *stable ids* (for prefix-trie block hashing). It never detokenizes. So the
// default here is the same deterministic 4-byte fold the Python lab falls back
// to when TOKENIZER_ID is unset, behind an interface so a real tokenizer can be
// swapped in without touching any caller.
package tokenize

import (
	"encoding/binary"
	"strings"
)

// Tokenizer converts text to token ids. Implementations must be safe for
// concurrent use: the gateway calls Encode from every request goroutine.
type Tokenizer interface {
	Encode(text string) []uint32
}

// Message is a minimal chat message, decoupled from any HTTP or vendor type.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// MessagesToText flattens a chat transcript the same way the Python lab does,
// so identical conversations produce identical prefixes and the trie behaves
// comparably across the two implementations.
func MessagesToText(msgs []Message) string {
	var b strings.Builder
	for i, m := range msgs {
		if i > 0 {
			b.WriteByte('\n')
		}
		role := m.Role
		if role == "" {
			role = "user"
		}
		b.WriteString(role)
		b.WriteString(": ")
		b.WriteString(m.Content)
	}
	return b.String()
}

// Fold4 is the fallback tokenizer: every 4 runes fold into one id. It is not a
// real tokenizer and makes no linguistic claim. It is deterministic, allocation
// -light, and monotonic in text length, which is all the gateway's counting and
// prefix matching require.
//
// Rune-based rather than byte-based to mirror Python's str slicing, so the same
// input yields the same token count in both implementations.
//
// Boundary behaviour: the final id folds a partial window, so a text and a
// longer text sharing a prefix agree on every id except that last one. This is
// what a real subword tokenizer does too, and it does not affect prefix
// matching because trie.BlockHashes discards trailing partial blocks.
type Fold4 struct{}

// Encode implements Tokenizer.
func (Fold4) Encode(text string) []uint32 {
	if text == "" {
		// Never return an empty id list: downstream code divides by token
		// counts, and an empty prompt is still one unit of work.
		return []uint32{0}
	}
	runes := []rune(text)
	ids := make([]uint32, 0, (len(runes)+3)/4)
	for i := 0; i < len(runes); i += 4 {
		end := min(i+4, len(runes))
		var buf [4]byte
		copy(buf[:], []byte(string(runes[i:end]))) // zero-padded, truncated to 4
		ids = append(ids, binary.LittleEndian.Uint32(buf[:])&0x7FFFFFFF)
	}
	return ids
}
