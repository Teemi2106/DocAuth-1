# TODO — OCR crash fix (EasyOCR)

- [x] Update `DocAuth/src/analysis/ocr.py`: validate image path, pre-validate with PIL, convert to safe temp PNG before calling EasyOCR.

- [x] Add clearer exceptions when decoding fails.

- [x] Update `DocAuth/app.py`: wrap OCR call in try/except and display Streamlit-friendly error message instead of crashing.

- [x] Re-run `streamlit run app.py` and test OCR with PNG/JPG, then TIFF/BMP.
