# 项目结构说明 / Project Structure

## 目录布局 / Directory Layout

```
CE泄露V3_ModelOnly_ActiveOnly/
├── scripts/                          # 实验脚本
│   ├── baseline/                     # 基线脚本 (不可修改，仅供对比)
│   │   ├── v3_2passKNN_refactored_celeak_modelonly_activeonly.py
│   │   └── PROVENANCE.md            # 溯源记录
│   └── experiments/                  # 实验分支脚本 (基于 baseline 修改)
├── logs/                             # 实验日志归档
│   └── c201_celeak_modelonly_activeonly/
│       └── experiment_summary.md     # 结果摘要
├── docs/                             # 项目文档
│   ├── PROJECT_STRUCTURE.md          # 本文件
│   └── methodology_notes.md          # 方法论笔记 (待创建)
├── configs/                          # 实验配置文件 (待创建)
├── sec/                              # LaTeX 论文各章节
├── pals/                             # PALS 论文模板 (子项目)
├── algorithms/                       # LaTeX 算法宏包
├── images/                           # 论文图片
├── main.tex                          # 论文主入口
├── main.bib                          # 参考文献
└── .gitignore
```

## 命名规范 / Naming Conventions

### 脚本命名
- 基线脚本保持原始文件名不变
- 实验分支脚本格式: `v3_celeak_<变体描述>_<日期>.py`

### 日志命名
- 日志文件名必须与生成它的脚本名称一致
- 格式: `<脚本基础名>_<数据集>_<关键参数>.log`

### 配置命名
- 格式: `config_<数据集>_<实验变体>.yaml`

## 实验流要求 / Experiment Workflow Constraints
- 所有对核心算法公式的修订，必须在此独立物理目录执行。
- 保证新目录内的结果变动不会与旧原子的结果进行混淆。
