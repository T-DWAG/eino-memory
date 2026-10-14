package memory

import (
	"context"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/embedding"
)

// fakeLLM 按调用顺序返回预设回复，用于离线测试需要 LLM 的策略。
type fakeLLM struct {
	replies []string
	calls   int
}

func (f *fakeLLM) GenerateText(_ context.Context, _, _ string) (string, error) {
	r := f.replies[f.calls%len(f.replies)]
	f.calls++
	return r, nil
}

// fakeEmbedder 把文本映射为"关键词命中"向量，模拟语义相似度。
type fakeEmbedder struct{ vocab []string }

func (e fakeEmbedder) EmbedStrings(_ context.Context, texts []string, _ ...embedding.Option) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i, t := range texts {
		v := make([]float64, len(e.vocab)+1)
		v[len(e.vocab)] = 0.01 // 避免零向量
		for j, w := range e.vocab {
			if strings.Contains(t, w) {
				v[j] = 1
			}
		}
		out[i] = v
	}
	return out, nil
}

var ctx = context.Background()

func TestSlidingWindowDropsOldest(t *testing.T) {
	m := NewSlidingWindowMemory(2)
	_ = m.AddMessage(ctx, "我叫max", "你好max")
	_ = m.AddMessage(ctx, "我用Python", "好的")
	_ = m.AddMessage(ctx, "我爱爬山", "不错")
	got, _ := m.GetContext(ctx, "")
	if strings.Contains(got, "max") {
		t.Fatalf("第一轮应被淘汰, got:\n%s", got)
	}
	if !strings.Contains(got, "爬山") || !strings.Contains(got, "Python") {
		t.Fatalf("应保留最近两轮, got:\n%s", got)
	}
}

func TestSummarizationConsolidates(t *testing.T) {
	llm := &fakeLLM{replies: []string{"公司Innovatech, 产品Helios"}}
	m := NewSummarizationMemory(llm, 4)
	_ = m.AddMessage(ctx, "a", "b")
	if llm.calls != 0 {
		t.Fatal("未达到阈值不应触发摘要")
	}
	_ = m.AddMessage(ctx, "c", "d")
	got, _ := m.GetContext(ctx, "")
	if llm.calls != 1 || !strings.Contains(got, "Innovatech") || len(m.buffer) != 0 {
		t.Fatalf("应触发摘要并清空缓冲, calls=%d ctx=%s", llm.calls, got)
	}
}

func TestRetrievalTopK(t *testing.T) {
	m := NewRetrievalMemory(fakeEmbedder{vocab: []string{"东京", "React", "Django"}}, 1)
	_ = m.AddMessage(ctx, "我想去东京", "好的")
	_ = m.AddMessage(ctx, "前端用React", "不错")
	got, err := m.GetContext(ctx, "东京有什么好玩")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "东京") || strings.Contains(got, "React") {
		t.Fatalf("应只检索到东京相关内容, got:\n%s", got)
	}
}

func TestMemoryAugmentedKeepsFacts(t *testing.T) {
	llm := &fakeLLM{replies: []string{"用户对花生严重过敏", noFactMarker, noFactMarker}}
	m := NewMemoryAugmentedMemory(llm, 1)
	_ = m.AddMessage(ctx, "我对花生过敏", "记住了")
	_ = m.AddMessage(ctx, "晚餐吃什么", "沙拉")
	_ = m.AddMessage(ctx, "甜点呢", "水果")
	got, _ := m.GetContext(ctx, "")
	if !strings.Contains(got, "花生") || len(m.memoryTokens) != 1 {
		t.Fatalf("过敏事实应保存在长期记忆中, got:\n%s", got)
	}
}

func TestHierarchicalPromotion(t *testing.T) {
	m := NewHierarchicalMemory(fakeEmbedder{vocab: []string{"ID", "天气"}}, 1, 1, nil)
	_ = m.AddMessage(ctx, "请记住我的用户ID是AX-7890", "好的")
	_ = m.AddMessage(ctx, "今天天气不错", "是的")
	got, _ := m.GetContext(ctx, "我的ID是什么")
	if !strings.Contains(got, "AX-7890") {
		t.Fatalf("用户ID应从长期记忆中检索到, got:\n%s", got)
	}
}

func TestGraphJSONAndTupleParsing(t *testing.T) {
	llm := &fakeLLM{replies: []string{
		"```json\n[{\"subject\":\"Clara\",\"relation\":\"works_for\",\"object\":\"FutureScape\"}]\n```",
		"[('Clara', 'main_project_is', 'Odyssey')]", // 兜底的元组格式
	}}
	m := NewGraphMemory(llm)
	_ = m.AddMessage(ctx, "Clara在FutureScape工作", "ok")
	_ = m.AddMessage(ctx, "Clara的项目是Odyssey", "ok")
	got, _ := m.GetContext(ctx, "告诉我关于Clara的项目。")
	for _, want := range []string{"Clara --[works_for]--> FutureScape", "Clara --[main_project_is]--> Odyssey"} {
		if !strings.Contains(got, want) {
			t.Fatalf("缺少关系 %q, got:\n%s", want, got)
		}
	}
}

func TestCompression(t *testing.T) {
	m := NewCompressionMemory(&fakeLLM{replies: []string{"会议地点: 大都会会议中心"}})
	_ = m.AddMessage(ctx, "场地定在大都会会议中心", "好的")
	got, _ := m.GetContext(ctx, "")
	if !strings.Contains(got, "- 会议地点") {
		t.Fatalf("got:\n%s", got)
	}
}

func TestOSMemoryPageInChinese(t *testing.T) {
	m := NewOSMemory(2)
	_ = m.AddMessage(ctx, "秘密发射代码是'Orion-Delta-7'。", "收到")
	_ = m.AddMessage(ctx, "发射的天气看起来很晴朗。", "好")
	_ = m.AddMessage(ctx, "发射窗口在0400 Zulu开启。", "好")
	got, _ := m.GetContext(ctx, "我需要确认发射代码。")
	if !strings.Contains(got, "(从轮次0换入)") || !strings.Contains(got, "Orion-Delta-7") {
		t.Fatalf("应从被动存储换入轮次0, got:\n%s", got)
	}
}
