# VS Code Extensions Recommended cho Rumor Detection Project

> Tài liệu này đề xuất các VS Code extension cần thiết dựa trên công nghệ sử dụng trong dự án `rumor-detection-project`.  
> Các extension được phân loại theo mức độ ưu tiên và mục đích sử dụng.

---

## 📋 Tổng quan công nghệ trong dự án

| Công nghệ | File / Module liên quan | Extension cần thiết |
|-----------|------------------------|---------------------|
| Python 3 | `main.py`, `preprocessing/`, `utils/`, `models/`, `knowledge_graph/` | Python, Pylance, Debugpy |
| Jupyter Notebook | `notebooks/*.ipynb` | Jupyter |
| Black / Flake8 | Theo `requirements.txt` | Black Formatter, Flake8 |
| Pandas / NumPy | `*.py` xử lý dữ liệu | Python Data Science pack |
| CSV | `data/processed/*.csv`, `final_metrics_table.csv` | Rainbow CSV |
| RDF / OWL / Turtle | `ontology/pheme_ontology_v1.ttl`, `knowledge_graph/build_kg.py` | RDF & SPARQL |
| NetworkX / Graph | `utils/graph_features.py`, `preprocessing/propagation_features.py` | (Python core) |
| Mermaid Diagram | `ontology_mermaid_class_diagram.md`, `ontology_mermaid_er_diagram.md` | Mermaid Preview |
| Graphviz | `ontology_diagram/` | Graphviz Preview |
| Markdown | `*.md`, `project_brain_bundle/` | Markdown All in One |
| Git | `.gitignore` | GitLens, Git Graph |
| TODO comments | Nhiều file có `# TODO:` | Todo Tree |

---

## ⭐ Mức độ ưu tiên 1: Bắt buộc (Core)

### 1. Python (`ms-python.python`)
- **Mục đích**: Hỗ trợ Python core: IntelliSense, linting, debugging, code navigation
- **Lý do**: Toàn bộ dự án viết bằng Python
- **Cài đặt**: `ext install ms-python.python`

### 2. Pylance (`ms-python.vscode-pylance`)
- **Mục đích**: Type checking nhanh, phân tích mã tĩnh, autocomplete chính xác
- **Lý do**: Dự án có nhiều type hints (`from typing import Dict, List, Optional, ...`)
- **Cài đặt**: `ext install ms-python.vscode-pylance`

### 3. Python Debugger (`ms-python.debugpy`)
- **Mục đích**: Debug Python code với breakpoints, watch variables
- **Lý do**: Debug pipeline preprocessing, knowledge graph xử lý dữ liệu lớn
- **Cài đặt**: `ext install ms-python.debugpy`

### 4. Jupyter (`ms-toolsai.jupyter`)
- **Mục đích**: Mở và chạy Jupyter Notebooks ngay trong VS Code
- **Lý do**: Dự án có 10+ notebook phân tích (pheme, graph, BERT fusion, v.v.)
- **Cài đặt**: `ext install ms-toolsai.jupyter`
- **File liên quan**: `notebooks/01_pheme_analysis.ipynb`, `notebooks/03_rumor_detection_baseline_fixed.ipynb`, `notebooks/05_bert_graph_fusion.ipynb`

### 5. Jupyter Notebook Renderers (`ms-toolsai.vscode-jupyter-notebook-renderers`)
- **Mục đích**: Render biểu đồ, bảng, output trong notebook
- **Lý do**: Các notebook hiển thị kết quả phân tích, metrics, charts
- **Cài đặt**: `ext install ms-toolsai.vscode-jupyter-notebook-renderers`

---

## ⭐⭐ Mức độ ưu tiên 2: Quan trọng (Essential)

### 6. Flake8 (`ms-python.flake8`)
- **Mục đích**: Linting Python code, phát hiện lỗi cú pháp, coding style
- **Lý do**: `requirements.txt` đã khai báo `flake8>=4.0.0` → dự án dùng flake8
- **Cài đặt**: `ext install ms-python.flake8`

### 7. Black Formatter (`ms-python.black-formatter`)
- **Mục đích**: Format code tự động theo chuẩn Black
- **Lý do**: `requirements.txt` đã khai báo `black>=21.0.0` → dự án dùng Black
- **Cài đặt**: `ext install ms-python.black-formatter`

### 8. isort (`ms-python.isort`)
- **Mục đích**: Sắp xếp import statements tự động
- **Lý do**: Giữ import consistent trong toàn bộ codebase
- **Cài đặt**: `ext install ms-python.isort`

### 9. Rainbow CSV (`mechatroner.rainbow-csv`)
- **Mục đích**: Highlight cột CSV với màu sắc, align columns
- **Lý do**: Dự án có nhiều file CSV:
  - `data/processed/pheme_features.csv`
  - `data/processed/pheme_clean.csv`
  - `final_metrics_table.csv`
  - `notebooks/final_metrics_table.csv`
- **Cài đặt**: `ext install mechatroner.rainbow-csv`

### 10. RDF & SPARQL Syntax Highlighting (`zazuko.rdf-vscode`)
- **Mục đích**: Highlight cú pháp cho file RDF, Turtle, SPARQL
- **Lý do**: Dự án có ontology file:
  - `ontology/pheme_ontology_v1.ttl` (157 dòng OWL ontology)
  - Knowledge graph output: `data/processed/pheme_kg.ttl`
- **Cài đặt**: `ext install zazuko.rdf-vscode`

---

## ⭐ Mức độ ưu tiên 3: Khuyến nghị (Recommended)

### 11. Markdown All in One (`yzhang.markdown-all-in-one`)
- **Mục đích**: Preview Markdown, table formatting, TOC generation
- **Lý do**: Dự án có nhiều file `.md` documentation
- **Cài đặt**: `ext install yzhang.markdown-all-in-one`

### 12. Mermaid Preview (`bierner.markdown-mermaid`)
- **Mục đích**: Hiển thị sơ đồ Mermaid trong Markdown preview
- **Lý do**: Dự án có sơ đồ ontology dạng Mermaid:
  - `ontology_mermaid_class_diagram.md`
  - `ontology_mermaid_er_diagram.md`
- **Cài đặt**: `ext install bierner.markdown-mermaid`

### 13. GitLens (`eamodio.gitlens`)
- **Mục đích**: Xem Git blame, history, code annotations
- **Lý do**: Quản lý version cho dự án nghiên cứu, track thay đổi
- **Cài đặt**: `ext install eamodio.gitlens`

### 14. Todo Tree (`Gruntfuggly.todo-tree`)
- **Mục đích**: Highlight và quản lý TODO/FIXME comments
- **Lý do**: Nhiều file có `# TODO:`, cần theo dõi các task chưa hoàn thành
  - Ví dụ: `preprocessing/features.py` có `# TODO: Implement feature extraction`
- **Cài đặt**: `ext install Gruntfuggly.todo-tree`

### 15. Path Intellisense (`christian-kohler.path-intellisense`)
- **Mục đích**: Autocomplete đường dẫn file khi import
- **Lý do**: Dự án có cấu trúc thư mục sâu, nhiều import với đường dẫn tương đối
- **Cài đặt**: `ext install christian-kohler.path-intellisense`

---

## 🛠 Mức độ ưu tiên 4: Tiện ích (Nice-to-have)

### 16. SPARQL Notebooks (`stardog-union.vscode-sparql-notebook`)
- **Mục đích**: Chạy SPARQL queries trong notebook
- **Lý do**: Query knowledge graph trực tiếp để kiểm tra dữ liệu RDF
- **Cài đặt**: `ext install stardog-union.vscode-sparql-notebook`

### 17. Graphviz Preview (`joaompinto.graphviz-preview`)
- **Mục đích**: Preview file Graphviz (.gv, .dot)
- **Lý do**: Dự án có thư mục `ontology_diagram/` chứa diagram
- **Cài đặt**: `ext install joaompinto.graphviz-preview`

### 18. Git Graph (`mhutchie.git-graph`)
- **Mục đích**: Visualize Git branches và commits
- **Lý do**: Dễ dàng xem lịch sử phát triển dự án
- **Cài đặt**: `ext install mhutchie.git-graph`

### 19. Jupyter Cell Tags (`ms-toolsai.vscode-jupyter-cell-tags`)
- **Mục đích**: Quản lý cell tags trong Jupyter notebooks
- **Lý do**: Hỗ trợ parameterized notebooks nếu cần
- **Cài đặt**: `ext install ms-toolsai.vscode-jupyter-cell-tags`

### 20. indent-rainbow (`oderwat.indent-rainbow`)
- **Mục đích**: Tô màu indent giúp đọc code dễ hơn
- **Lý do**: Python dựa vào indent, đặc biệt khi code lồng nhiều cấp
- **Cài đặt**: `ext install oderwat.indent-rainbow`

---

## 📥 Cài đặt hàng loạt

Mở VS Code Command Palette (`Ctrl+Shift+P`) và chạy: `>Extensions: Show Recommended Extensions`

Hoặc copy file `.vscode/extensions.json` vào dự án và VS Code sẽ tự động gợi ý cài.

Để cài tất cả extension được recommend:
1. Mở Command Palette (`Ctrl+Shift+P`)
2. Gõ: `Extensions: Show Recommended Extensions`
3. Nhấn "Install Workspace Recommended Extensions"

---

## 🔧 Cấu hình VS Code settings khuyên dùng

Tạo file `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "none",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "isort.args": ["--profile", "black"],
  "black-formatter.args": ["--line-length", "120"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.ipynb_checkpoints": true,
    "**/.pytest_cache": true
  },
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "editor.rulers": [88, 120],
  "todo-tree.general.tags": ["TODO", "FIXME", "HACK", "BUG", "NOTE"],
  "todo-tree.highlights.defaultHighlight": {
    "icon": "check",
    "type": "text"
  }
}
```

---

## 📊 Tóm tắt nhanh

| # | Extension | ID | Priority | Lý do chính |
|---|-----------|-----|----------|-------------|
| 1 | Python | `ms-python.python` | ⭐⭐⭐ | Core language |
| 2 | Pylance | `ms-python.vscode-pylance` | ⭐⭐⭐ | Type checking |
| 3 | Debugpy | `ms-python.debugpy` | ⭐⭐⭐ | Debug |
| 4 | Jupyter | `ms-toolsai.jupyter` | ⭐⭐⭐ | Notebooks (10+ files) |
| 5 | Jupyter Renderers | `ms-toolsai.vscode-jupyter-notebook-renderers` | ⭐⭐⭐ | Render output |
| 6 | Flake8 | `ms-python.flake8` | ⭐⭐ | Linting |
| 7 | Black | `ms-python.black-formatter` | ⭐⭐ | Formatting |
| 8 | isort | `ms-python.isort` | ⭐⭐ | Import sorting |
| 9 | Rainbow CSV | `mechatroner.rainbow-csv` | ⭐⭐ | CSV files |
| 10 | RDF & SPARQL | `zazuko.rdf-vscode` | ⭐⭐ | Ontology `.ttl` |
| 11 | Markdown AIO | `yzhang.markdown-all-in-one` | ⭐ | Documentation |
| 12 | Mermaid | `bierner.markdown-mermaid` | ⭐ | Mermaid diagrams |
| 13 | GitLens | `eamodio.gitlens` | ⭐ | Version control |
| 14 | Todo Tree | `Gruntfuggly.todo-tree` | ⭐ | Track TODOs |
| 15 | Path Intellisense | `christian-kohler.path-intellisense` | ⭐ | File paths |
| 16 | SPARQL Notebooks | `stardog-union.vscode-sparql-notebook` | 🛠 | SPARQL queries |
| 17 | Graphviz | `joaompinto.graphviz-preview` | 🛠 | Graphviz diagrams |
| 18 | Git Graph | `mhutchie.git-graph` | 🛠 | Git visualization |
| 19 | Jupyter Cell Tags | `ms-toolsai.vscode-jupyter-cell-tags` | 🛠 | Notebook cell tags |
| 20 | indent-rainbow | `oderwat.indent-rainbow` | 🛠 | Indent coloring |