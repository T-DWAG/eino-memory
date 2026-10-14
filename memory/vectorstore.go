package memory

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/indexer"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
)

// ============================================================================
// InMemoryVectorStore：基于内存的向量库
// ----------------------------------------------------------------------------
// 暴力精确搜索：把查询向量与库中每个向量逐一比对。
//
// 它同时实现了 eino 的两个标准组件接口：
//   - indexer.Indexer     （Store：向量化并写入文档）
//   - retriever.Retriever （Retrieve：按语义相似度检索文档）
//
// 因此它可以直接被 compose.Chain / Graph 编排（AppendIndexer / AppendRetriever）。
// 生产环境只需把它替换成 eino-ext 中的 Milvus / Redis / ES8 等实现，
// 记忆策略代码无需改动。
//
// 相似度：使用余弦相似度（分数越大越相似），比 L2 距离更适合文本 embedding。
// ============================================================================

var (
	_ indexer.Indexer     = (*InMemoryVectorStore)(nil)
	_ retriever.Retriever = (*InMemoryVectorStore)(nil)
)

// InMemoryVectorStore 线程安全的内存向量库。
type InMemoryVectorStore struct {
	mu       sync.RWMutex
	embedder embedding.Embedder
	topK     int
	docs     []*schema.Document
	vectors  [][]float64
}

// NewInMemoryVectorStore 创建向量库。topK 为默认检索条数。
func NewInMemoryVectorStore(embedder embedding.Embedder, topK int) *InMemoryVectorStore {
	if topK <= 0 {
		topK = 2
	}
	return &InMemoryVectorStore{embedder: embedder, topK: topK}
}

// Store 实现 indexer.Indexer：批量向量化文档并写入内存。
func (s *InMemoryVectorStore) Store(ctx context.Context, docs []*schema.Document, opts ...indexer.Option) ([]string, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	// 支持通过 indexer.WithEmbedding 临时覆盖 embedder（eino 通用 Option 约定）
	o := indexer.GetCommonOptions(&indexer.Options{Embedding: s.embedder}, opts...)
	if o.Embedding == nil {
		return nil, errors.New("vector store: 未配置 embedder")
	}

	texts := make([]string, len(docs))
	for i, d := range docs {
		texts[i] = d.Content
	}
	vecs, err := o.Embedding.EmbedStrings(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("vector store: 向量化失败: %w", err)
	}
	if len(vecs) != len(docs) {
		return nil, fmt.Errorf("vector store: 向量数量(%d)与文档数量(%d)不一致", len(vecs), len(docs))
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	ids := make([]string, len(docs))
	for i, d := range docs {
		if d.ID == "" {
			d.ID = fmt.Sprintf("doc-%d", len(s.docs))
		}
		ids[i] = d.ID
		s.docs = append(s.docs, d)
		s.vectors = append(s.vectors, vecs[i])
	}
	return ids, nil
}

// Retrieve 实现 retriever.Retriever：返回与 query 最相似的 TopK 文档（附带分数）。
func (s *InMemoryVectorStore) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	topK := s.topK
	o := retriever.GetCommonOptions(&retriever.Options{TopK: &topK, Embedding: s.embedder}, opts...)

	s.mu.RLock()
	empty := len(s.docs) == 0
	s.mu.RUnlock()
	if empty {
		return nil, nil
	}

	qv, err := o.Embedding.EmbedStrings(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("vector store: 查询向量化失败: %w", err)
	}
	if len(qv) == 0 {
		return nil, errors.New("vector store: 查询向量为空")
	}

	type scored struct {
		doc   *schema.Document
		score float64
	}
	s.mu.RLock()
	results := make([]scored, 0, len(s.docs))
	for i, v := range s.vectors {
		results = append(results, scored{doc: s.docs[i], score: cosine(qv[0], v)})
	}
	s.mu.RUnlock()

	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })

	k := *o.TopK
	if k > len(results) {
		k = len(results)
	}
	out := make([]*schema.Document, 0, k)
	for _, r := range results[:k] {
		// 可选的分数阈值过滤（retriever.WithScoreThreshold）
		if o.ScoreThreshold != nil && r.score < *o.ScoreThreshold {
			continue
		}
		// 复制文档并深拷贝 MetaData 再写分数：WithScore 会写 map，
		// 浅拷贝会共享同一个 map，并发检索时产生数据竞争。
		cp := *r.doc
		if r.doc.MetaData != nil {
			cp.MetaData = make(map[string]any, len(r.doc.MetaData)+1)
			for k, v := range r.doc.MetaData {
				cp.MetaData[k] = v
			}
		}
		out = append(out, cp.WithScore(r.score))
	}
	return out, nil
}

// Len 返回已存储的文档数量。
func (s *InMemoryVectorStore) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.docs)
}

// Reset 清空所有文档和向量。
func (s *InMemoryVectorStore) Reset() {
	s.mu.Lock()
	s.docs, s.vectors = nil, nil
	s.mu.Unlock()
}

// cosine 计算两个向量的余弦相似度；维度不一致或零向量返回 0。
func cosine(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
