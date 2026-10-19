package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

// Exercise the actual SDK serializer after genRequest has resolved bound and
// per-request tools. A nil request override inherits bound tools; an explicit
// empty override removes them. No provider or model endpoint is contacted.
func TestEffectiveToolChoiceWire(t *testing.T) {
	tool := func(name string) *schema.ToolInfo {
		return &schema.ToolInfo{Name: name, Desc: "offline fixture", ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{"query": {Type: schema.String, Required: true}})}
	}
	alpha, beta := tool("alpha"), tool("beta")
	both := []*schema.ToolInfo{alpha, beta}
	specific := func(name string) any {
		return map[string]any{"type": "function", "function": map[string]any{"name": name}}
	}
	cases := []struct {
		name        string
		bound       bool
		boundForced bool
		options     []model.Option
		tools       int
		choice      any
		wantError   bool
		forbidden   bool
	}{
		{name: "no_tools_default"},
		{name: "no_tools_forbidden", options: []model.Option{model.WithToolChoice(schema.ToolChoiceForbidden)}, forbidden: true},
		{name: "explicit_empty_tools_forbidden", options: []model.Option{model.WithTools([]*schema.ToolInfo{}), model.WithToolChoice(schema.ToolChoiceForbidden)}, forbidden: true},
		{name: "request_tools_forbidden", options: []model.Option{model.WithTools(both), model.WithToolChoice(schema.ToolChoiceForbidden)}, tools: 2, choice: "none", forbidden: true},
		{name: "bound_tools_forbidden", bound: true, options: []model.Option{model.WithToolChoice(schema.ToolChoiceForbidden)}, tools: 2, choice: "none", forbidden: true},
		{name: "bound_forced_tools_forbidden", boundForced: true, options: []model.Option{model.WithToolChoice(schema.ToolChoiceForbidden)}, tools: 2, choice: "none", forbidden: true},
		{name: "bound_tools_explicit_empty_forbidden", bound: true, options: []model.Option{model.WithTools([]*schema.ToolInfo{}), model.WithToolChoice(schema.ToolChoiceForbidden)}, forbidden: true},
		{name: "bound_forced_explicit_empty_forbidden", boundForced: true, options: []model.Option{model.WithTools([]*schema.ToolInfo{}), model.WithToolChoice(schema.ToolChoiceForbidden)}, forbidden: true},
		// Auto behavior is intentionally unchanged by the empty-Forbidden fix.
		{name: "no_tools_auto_unchanged", options: []model.Option{model.WithToolChoice(schema.ToolChoiceAllowed)}, choice: "auto"},
		{name: "request_tools_auto", options: []model.Option{model.WithTools(both), model.WithToolChoice(schema.ToolChoiceAllowed)}, tools: 2, choice: "auto"},
		{name: "bound_tools_inherited_auto", bound: true, tools: 2, choice: "auto"},
		{name: "bound_tools_explicit_empty_auto_unchanged", bound: true, options: []model.Option{model.WithTools([]*schema.ToolInfo{})}, choice: "auto"},
		{name: "no_tools_required_rejected", options: []model.Option{model.WithToolChoice(schema.ToolChoiceForced)}, wantError: true},
		{name: "bound_forced_explicit_empty_required_rejected", boundForced: true, options: []model.Option{model.WithTools([]*schema.ToolInfo{})}, wantError: true},
		{name: "request_multiple_tools_required", options: []model.Option{model.WithTools(both), model.WithToolChoice(schema.ToolChoiceForced)}, tools: 2, choice: "required"},
		{name: "bound_forced_multiple_tools_required", boundForced: true, tools: 2, choice: "required"},
		{name: "request_one_tool_specific", options: []model.Option{model.WithTools([]*schema.ToolInfo{alpha}), model.WithToolChoice(schema.ToolChoiceForced)}, tools: 1, choice: specific("alpha")},
		{name: "request_explicit_specific", options: []model.Option{model.WithTools(both), model.WithToolChoice(schema.ToolChoiceForced, "beta")}, tools: 2, choice: specific("beta")},
		{name: "bound_tools_explicit_specific", bound: true, options: []model.Option{model.WithToolChoice(schema.ToolChoiceForced, "beta")}, tools: 2, choice: specific("beta")},
		{name: "missing_specific_rejected", options: []model.Option{model.WithTools(both), model.WithToolChoice(schema.ToolChoiceForced, "missing")}, wantError: true},
		{name: "no_tools_specific_rejected", options: []model.Option{model.WithToolChoice(schema.ToolChoiceForced, "alpha")}, wantError: true},
		{name: "missing_allowed_auto_rejected", options: []model.Option{model.WithToolChoice(schema.ToolChoiceAllowed, "missing")}, wantError: true},
	}
	for _, streaming := range []bool{false, true} {
		mode := "generate"
		if streaming {
			mode = "stream"
		}
		for _, test := range cases {
			t.Run(mode+"/"+test.name, func(t *testing.T) {
				var calls atomic.Int32
				bodies := make(chan map[string]any, 1)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls.Add(1)
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error("invalid SDK JSON", err)
						w.WriteHeader(400)
						return
					}
					bodies <- body
					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						io.WriteString(w, "data: {\"id\":\"offline\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"offline-final\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
					} else {
						w.Header().Set("Content-Type", "application/json")
						io.WriteString(w, `{"id":"offline","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"offline-final"},"finish_reason":"stop"}]}`)
					}
				}))
				defer server.Close()
				client, err := NewClient(t.Context(), &Config{Model: "offline-only", APIKey: "offline-placeholder", BaseURL: server.URL + "/v1", HTTPClient: server.Client()})
				if err != nil {
					t.Fatal(err)
				}
				if test.bound {
					if err = client.BindTools(both); err != nil {
						t.Fatal(err)
					}
				}
				if test.boundForced {
					if err = client.BindForcedTools(both); err != nil {
						t.Fatal(err)
					}
				}
				messages := []*schema.Message{{Role: schema.User, Content: "offline protocol fixture"}}
				if test.forbidden {
					_, callback, _, _, e := client.genRequest(t.Context(), messages, test.options...)
					if e != nil {
						t.Fatal(e)
					}
					if callback.ToolChoice == nil || *callback.ToolChoice != schema.ToolChoiceForbidden {
						t.Fatal("business Forbidden choice was weakened")
					}
				}
				if streaming {
					var reader *schema.StreamReader[*schema.Message]
					reader, err = client.Stream(t.Context(), messages, test.options...)
					if err == nil {
						for {
							_, e := reader.Recv()
							if e == io.EOF {
								break
							}
							if e != nil {
								err = e
								break
							}
						}
						reader.Close()
					}
				} else {
					_, err = client.Generate(t.Context(), messages, test.options...)
				}
				if test.wantError {
					if err == nil || calls.Load() != 0 {
						t.Fatalf("invalid Required/specific request escaped: err=%v calls=%d", err, calls.Load())
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				if calls.Load() != 1 {
					t.Fatalf("HTTP calls=%d", calls.Load())
				}
				body := <-bodies
				tools := 0
				if values, ok := body["tools"].([]any); ok {
					tools = len(values)
				}
				if tools != test.tools {
					t.Fatalf("effective tools=%d want=%d", tools, test.tools)
				}
				choice, present := body["tool_choice"]
				if !reflect.DeepEqual(choice, test.choice) {
					t.Fatalf("serialized tool_choice=%v want=%v", choice, test.choice)
				}
				if test.choice == nil && present {
					t.Fatal("tool_choice must be omitted, not null")
				}
				if (test.bound || test.boundForced) && len(client.rawTools) != 2 {
					t.Fatal("per-request override mutated bound tools")
				}
			})
		}
	}
}
