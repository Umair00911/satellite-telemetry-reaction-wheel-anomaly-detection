try:
    with open('requirements.txt', 'r', encoding='utf-16') as f:
        print(f.read())
except Exception as e:
    with open('requirements.txt', 'r', encoding='utf-8', errors='ignore') as f:
        print(f.read())
