"""生成合成样本并写出 HTML / PDF 诊断报告。"""

from factorlab.cli import main

if __name__ == "__main__":
    main(["--demo", "--out", "output/demo"])
