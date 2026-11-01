"""Add continuous line numbering to every section of a .docx (Usage: python docx_line_numbers.py in.docx out.docx)"""
import re, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(src)
zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = zin.read(it.filename)
    if it.filename == "word/document.xml":
        x = data.decode("utf-8")
        ln = '<w:lnNumType w:countBy="1" w:distance="284" w:restart="continuous"/>'
        x = re.sub(r"<w:sectPr\s*/>", "<w:sectPr>" + ln + "</w:sectPr>", x)
        x = re.sub(r"(<w:sectPr(?:\s[^>]*)?>)(?!<w:lnNumType)", r"\1" + ln, x)
        data = x.encode("utf-8")
    zout.writestr(it, data)
zout.close()
