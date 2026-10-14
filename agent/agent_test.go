package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"

	"github.com/T-DWAG/eino-memory/memory"
)

// echoModel 是一个假的 eino ChatModel：把收到的 user 消息原样返回，
// 用于验证 Chain 中模板渲染与记忆注入是否正确。
type echoModel struct{ last []*schema.Message }

func (e *echoModel) Generate(_ context.Context, in []*schema.Message, _ ...model.Option) (*schema.Message, error) {
	e.last = in
	return schema.AssistantMessage("echo:"+in[len(in)-1].Content, nil), nil
}

func (e *echoModel) Stream(context.Context, []*schema.Message, ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	panic("not used")
}

func TestAgentChainInjectsMemory(t *testing.T) {
	ctx := context.Background()
	m := &echoModel{}
	mem := memory.NewSequentialMemory()
	// 系统提示词里含花括号，验证不会被 FString 当成占位符
	a, err := New(ctx, m, mem, WithSystemPrompt("你是助手 {不是变量}"))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := a.Chat(ctx, "我叫max"); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Chat(ctx, "我叫什么"); err != nil {
		t.Fatal(err)
	}

	if got := m.last[0].Content; got != "你是助手 {不是变量}" {
		t.Fatalf("system prompt 渲染错误: %q", got)
	}
	user := m.last[1].Content
	if !strings.Contains(user, "### MEMORY CONTEXT\nUser: 我叫max") || !strings.HasSuffix(user, "### CURRENT REQUEST\n我叫什么") {
		t.Fatalf("user prompt 渲染错误:\n%s", user)
	}
}
