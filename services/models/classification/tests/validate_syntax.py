import ast, sys
with open("/home/nourmaatouk/Solana-AI-Trading-System/services/models/classification/07_prepare_no_leak_data.py") as f:
    code = f.read()
try:
    ast.parse(code)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax Error: {e}")
    sys.exit(1)
