// 命令 demo 演示 9 种记忆策略的对话效果。
//
// 用法：
//
//	export OPENAI_API_KEY=sk-xxx
//	go run ./cmd/demo -s sliding      # 运行单个策略
//	go run ./cmd/demo -s all          # 依次运行全部策略
//
// 可选策略：sequential | sliding | summary | retrieval | augmented |
//
//	hierarchical | graph | compression | os | all
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/T-DWAG/eino-memory/agent"
	"github.com/T-DWAG/eino-memory/llm"
	"github.com/T-DWAG/eino-memory/memory"
)

// scenario 描述一个策略的演示：如何构造记忆 + 一组对话输入。
type scenario struct {
	build  func(c *llm.Client) memory.Strategy
	inputs []string
}

// scenarios 定义每种策略的演示对话。
var scenarios = map[string]scenario{
	"sequential": {
		build:  func(*llm.Client) memory.Strategy { return memory.NewSequentialMemory() },
		inputs: []string{"你好啊，我叫T-DWAG", "我对太空探索非常感兴趣", "我告诉你的第一件事情是什么"},
	},
	"sliding": {
		// 窗口 = 2：第 3 轮时第 1 轮（名字）被淘汰，第 4 轮问名字大概率答不上来
		build:  func(*llm.Client) memory.Strategy { return memory.NewSlidingWindowMemory(2) },
		inputs: []string{"我的名字叫T-DWAG", "我一般用Python写代码", "我的兴趣爱好是爬山", "我叫什么名字呢"},
	},
	"summary": {
		build: func(c *llm.Client) memory.Strategy { return memory.NewSummarizationMemory(c, 4) },
		inputs: []string{
			"我要开一家新公司，名字叫 Innovatech，主打可持续能源",
			"我们的第一款产品是一块智能太阳能板，内部代号叫‘Project Helios’.",
			"市场预算定在 5 万美元",
			"我的公司叫什么？第一款产品又叫什么?",
		},
	},
	"retrieval": {
		build: func(c *llm.Client) memory.Strategy { return memory.NewRetrievalMemory(c.Embedder, 2) },
		inputs: []string{
			"我计划明年春天去日本度假。",
			"对于我的软件项目,我使用React框架作为前端。",
			"我想在旅行中访问东京和京都。",
			"项目的后端将使用Django构建。",
			"我计划在度假时访问哪些城市?",
		},
	},
	"augmented": {
		build: func(c *llm.Client) memory.Strategy { return memory.NewMemoryAugmentedMemory(c, 2) },
		inputs: []string{
			"请记住这个以便将来所有交互:我对花生严重过敏。",
			"好的,我们来谈谈食谱。今晚晚餐有什么好主意?",
			"听起来不错。有什么甜点选择吗?",
			"你能推荐一个泰式绿咖喱食谱吗?请确保对我安全。",
		},
	},
	"hierarchical": {
		build: func(c *llm.Client) memory.Strategy { return memory.NewHierarchicalMemory(c.Embedder, 2, 2, nil) },
		inputs: []string{
			"请记住我的用户ID是AX-7890。",
			"我们来聊聊天气。今天阳光明媚。",
			"我计划稍后去散步。",
			"我需要登录我的账户,你能提醒我的ID吗?",
		},
	},
	"graph": {
		build: func(c *llm.Client) memory.Strategy { return memory.NewGraphMemory(c) },
		inputs: []string{
			"一个名叫Clara的人在一家叫'FutureScape'的公司工作。",
			"FutureScape总部位于柏林。",
			"Clara的主要项目名为'Odyssey'。",
			"告诉我关于Clara的项目。",
		},
	},
	"compression": {
		build: func(c *llm.Client) memory.Strategy { return memory.NewCompressionMemory(c) },
		inputs: []string{
			"好的,我已经决定了会议的场地。将在'大都会会议中心'举行。",
			"日期确定为2025年10月26日。",
			"你能总结一下会议计划的关键细节吗?",
		},
	},
	"os": {
		build: func(*llm.Client) memory.Strategy { return memory.NewOSMemory(2) },
		inputs: []string{
			"秘密发射代码是'Orion-Delta-7'。",
			"发射的天气看起来很晴朗。",
			"发射窗口在0400 Zulu开启。", // 这一轮会把"秘密代码"换出到被动存储
			"我需要确认发射代码。",        // 命中"发射""代码"，触发缺页换入
		},
	},
}

// order 运行 all 时的顺序。
var order = []string{"sequential", "sliding", "summary", "retrieval", "augmented", "hierarchical", "graph", "compression", "os"}

func main() {
	strategy := flag.String("s", "sliding", "记忆策略: "+strings.Join(order, " | ")+" | all")
	flag.Parse()

	ctx := context.Background()
	client, err := llm.NewClient(ctx, llm.ConfigFromEnv())
	if err != nil {
		log.Fatal(err)
	}

	names := []string{*strategy}
	if *strategy == "all" {
		names = order
	}
	for _, name := range names {
		if err := run(ctx, client, name); err != nil {
			fmt.Fprintf(os.Stderr, "[%s] 运行失败: %v\n", name, err)
			os.Exit(1)
		}
	}
}

// run 构建指定策略的 Agent 并依次发送演示对话。
func run(ctx context.Context, client *llm.Client, name string) error {
	sc, ok := scenarios[name]
	if !ok {
		return fmt.Errorf("未知策略 %q", name)
	}
	fmt.Printf("\n\n######## 策略: %s ########\n", name)

	mem := sc.build(client)
	a, err := agent.New(ctx, client.Chat, mem)
	if err != nil {
		return err
	}
	defer mem.Clear()

	for _, in := range sc.inputs {
		if _, err := a.Chat(ctx, in); err != nil {
			return err
		}
	}
	return nil
}
