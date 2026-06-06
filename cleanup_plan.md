# Cleanup Plan — Rumor Detection Project

**Date:** June 6, 2026  
**Based on:** REPOSITORY_AUDIT_REPORT.md  
**Confidence threshold:** >= 95%

---

## Files Proposed for Deletion

| # | File/Directory | Reason | Confidence | Impact if Deleted |
|---|---|---|---|---|
| 1 | `results/test.txt` | File chỉ chứa chữ "hello". Không liên quan đến pipeline, báo cáo, hay bất kỳ thành phần nào của đồ án. File test tạm thời. | 100% | Không ảnh hưởng. Pipeline không đọc file này. |
| 2 | `preprocessing/features.py` | File rỗng, chỉ có comment TODO. Chưa bao giờ được implement. Không được import bởi bất kỳ module nào trong project. | 100% | Không ảnh hưởng. File chưa bao giờ được sử dụng. |
| 3 | `config/config.py` | File rỗng, chỉ có 1 dòng comment. Không chứa cấu hình thực tế. Không được import bởi bất kỳ module nào. | 100% | Không ảnh hưởng. Cấu hình được hardcode trong từng script/notebook. |
| 4 | `models/__init__.py` | File rỗng, chỉ có comment. Các model được huấn luyện trực tiếp trong notebooks/scripts, không có module model riêng biệt. | 100% | Không ảnh hưởng. Không có model class nào được định nghĩa trong thư mục models/. |
| 5 | `notebooks/.ipynb_checkpoints/` | Thư mục cache tự động của Jupyter Notebook. Chứa file `ablation_study-checkpoint.ipynb` — bản sao dự phòng tự động của `ablation_study.ipynb`. Có thể tạo lại khi mở notebook. | 100% | Không ảnh hưởng. Jupyter sẽ tự động tạo lại khi cần. |

---

## Summary

| Metric | Count |
|--------|-------|
| Files to delete | 4 files |
| Directories to delete | 1 directory (ipynb_checkpoints) |
| Total items | 5 |
| Estimated space freed | ~2-5 KB (negligible) |
| Files preserved | ~85+ |

## Verification Checklist (After Deletion)

- [ ] `preprocessing/__init__.py` vẫn tồn tại (module preprocessing vẫn import được)
- [ ] Pipeline vẫn chạy được (first_baseline.py, run_ablation.py)
- [ ] Ontology v1 + v2 còn nguyên
- [ ] KG v1 + v2 còn nguyên
- [ ] Tất cả notebooks vẫn mở được
- [ ] Feature extraction còn nguyên
- [ ] Metrics và figures còn nguyên