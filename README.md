# LLMLogParsing

日志解析 × 模板匹配 × 变量抽取 · 基于 Drain3 与大规模日志样本自动聚类分析



## 📌 项目简介

本项目基于 [Drain3](https://github.com/logpai/drain3) 实现了对结构化日志的高性能语义模板挖掘，并结合正则表达式自动完成模板变量提取。我们从 3GB 级别的大型日志文件集中自动解析、聚类、可视化，并导出 CSV 分析结果，适用于日志运维分析、根因定位、故障溯源等场景。



## 📂 数据结构

- 项目中的日志主目录为：`2025_5_8_log_all/`
- 文件夹中包含 24 个 `.log/.txt/.out` 日志文件
- 总计提取出 **362,034 条结构化日志条目**



## 🔧 技术栈与依赖

- Python 3.8+
- drain3
- pandas / matplotlib / seaborn
- 正则表达式（用于变量提取）
- 可选：Jupyter Notebook 进行分析展示



## 🧠 模板抽取流程（核心步骤）

1. 递归扫描日志文件，提取有效行内容
2. 使用 `Drain3` 进行模板聚类（参数自定义）
3. 将所有 `<DATE>`、`<P_ID>`、`<T_ID>`、数字位置归一化为 `<*>`
4. 使用正则方式从原始日志中反向提取变量
5. 统计每个模板的匹配频次，输出覆盖率分析
6. 支持图表展示 Top-N 模板及累计覆盖曲线



## 🧪 样例输出

```text
🔍 示例模板匹配结果 (前5条):

[Log ID] 0
Original : 2025-04-30 00:00:01.319 [WARNING] database P0000009826 T0000000000000034013  socket_err_should_retry errno:38
Template : <DATE> [WARNING] database <P_ID> <*> socket_err_should_retry <*>
Variables: ['2025-04-30 00:00:01.319', 'P0000009826', 'T0000000000000034013', 'errno:38']
```
