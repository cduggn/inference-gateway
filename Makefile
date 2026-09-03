BIN := bin
GO  ?= go

.PHONY: all build test race lint fmt vet tidy clean run-gateway run-agent

all: fmt vet test build

build:
	$(GO) build -o $(BIN)/gateway ./cmd/gateway
	$(GO) build -o $(BIN)/agent   ./cmd/agent

test:
	$(GO) test ./...

race:
	$(GO) test -race ./...

cover:
	$(GO) test -coverprofile=coverage.out ./...
	$(GO) tool cover -func=coverage.out | tail -1

fmt:
	$(GO) fmt ./...

vet:
	$(GO) vet ./...

tidy:
	$(GO) mod tidy

clean:
	rm -rf $(BIN) coverage.out

# Run the gateway with everything switched on.
run-gateway: build
	$(BIN)/gateway --preset full --p2c

# Drive it with the agent pipeline.
run-agent: build
	$(BIN)/agent --repeat 5
