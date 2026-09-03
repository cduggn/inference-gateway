package tokenize

import "testing"

func TestMessagesToTextIsStable(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "be brief"},
		{Content: "hello"}, // empty role defaults to user
	}
	want := "system: be brief\nuser: hello"
	if got := MessagesToText(msgs); got != want {
		t.Errorf("MessagesToText() = %q, want %q", got, want)
	}
}

func TestEncodeIsDeterministicAndPrefixStable(t *testing.T) {
	tok := Fold4{}
	a := tok.Encode("the quick brown fox jumps")
	b := tok.Encode("the quick brown fox jumps")
	if len(a) != len(b) {
		t.Fatalf("same input produced %d and %d ids", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("id %d differs between runs: %d vs %d", i, a[i], b[i])
		}
	}

	// A shared text prefix must produce a shared id prefix, or prefix-trie
	// matching would never fire.
	//
	// All ids but the last: the final id folds a partial window, so extending
	// the text fills that window with different characters and changes it. Real
	// subword tokenizers behave the same way at the boundary, and the trie is
	// unaffected because it drops trailing partial blocks anyway.
	ext := tok.Encode("the quick brown fox jumps over the lazy dog")
	for i := range len(a) - 1 {
		if ext[i] != a[i] {
			t.Fatalf("extended text diverged at id %d, before the final partial window", i)
		}
	}
}

func TestEncodeEmptyStillCountsAsWork(t *testing.T) {
	if got := (Fold4{}).Encode(""); len(got) != 1 {
		t.Errorf("empty input produced %d ids, want 1", len(got))
	}
}
