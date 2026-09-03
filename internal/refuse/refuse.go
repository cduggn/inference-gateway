// Package refuse enumerates the reasons the gateway declines a request and the
// HTTP semantics that go with each one.
//
// The Python original scattered these as bare strings across app.py plus a
// RETRY_AFTER_S map in config.py. Making them a named type means the compiler
// catches a typo'd reason and every reason is forced to declare its own status
// and Retry-After hint in one place.
package refuse

import (
	"net/http"
	"time"
)

// Reason identifies why a request was refused. The string values are part of
// the wire contract: they appear verbatim in the JSON error body.
type Reason string

const (
	// Quota: the tenant's token bucket is empty.
	Quota Reason = "quota"

	// Admission-control reasons, evaluated in the order listed.
	NoSignal           Reason = "no_signal"
	KVPressure         Reason = "kv_pressure"
	QueueDepth         Reason = "queue_depth"
	NoHeadroom         Reason = "no_headroom"
	DeadlineUnmeetable Reason = "deadline_unmeetable"

	// Queue reasons.
	QueueFull       Reason = "queue_full"
	ExpiredInQueue  Reason = "expired_in_queue"
	ClientCancelled Reason = "client_cancelled"
)

// HTTPStatus is the status code to serve for this reason. Quota is a 429 so
// clients can distinguish "you are over your own budget" (retry later, same
// size) from a 503 "the fleet cannot take this right now" (retry later, maybe
// smaller).
func (r Reason) HTTPStatus() int {
	switch r {
	case Quota:
		return http.StatusTooManyRequests
	case ClientCancelled:
		// 499 is nginx's non-standard "client closed request". Nothing is
		// written to a disconnected client anyway; this only shapes the log.
		return 499
	default:
		return http.StatusServiceUnavailable
	}
}

// RetryAfter is the hint sent in the Retry-After header. Pressure that needs
// the fleet to drain KV cache gets a longer backoff than a transient queue.
func (r Reason) RetryAfter() time.Duration {
	switch r {
	case KVPressure, NoHeadroom:
		return 2 * time.Second
	default:
		return time.Second
	}
}

// Error lets a Reason be returned as an error value rather than thrown as an
// exception, which is how the Python version signalled QueueFull/Expired.
type Error struct {
	Reason Reason
}

func (e *Error) Error() string { return string(e.Reason) }

// Err wraps a Reason as an error.
func Err(r Reason) error { return &Error{Reason: r} }
