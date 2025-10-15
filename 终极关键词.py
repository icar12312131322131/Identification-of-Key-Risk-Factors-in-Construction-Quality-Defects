import re
import pandas as pd

# --------------------------
# 1. Read text file
# --------------------------
file_path = "your_input_file.txt"  # Replace with your file path
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# --------------------------
# 2. Preprocess text (character replacement and cleanup)
# --------------------------
content = content.replace('【', '（').replace('】', '）').replace(' ', '').replace('\t', '')
content = re.sub(r'[﹄﹃“”]', lambda m: {'﹄':'）','﹃':'（','“':'“','”':'”'}.get(m.group(), ''), content)

# --------------------------
# 3. Extract quality issue titles
# --------------------------
title_pattern = re.compile(
    r'\n(\d+\.\d+\.\d+)\s*([^\n]+?)\s*(\n|$)\s*1\s*[\.。]\s*现象', 
    re.DOTALL
)
titles = title_pattern.findall(content)
quality_issues = [f"{num} {name.strip()}" for num, name, _ in titles]

# --------------------------
# 4. Extract cause analysis sections
# --------------------------
cause_pattern = re.compile(
    r'原因分析\s*\n((?:.(?!\s*(?:原因分析|防治措施|预防措施|\d+[\.。）》]|$))*.|\n)*?)'  
    r'(?=\s*\n\s*(?:防治措施|预防措施|\d+[\.。）》]|$))', 
    re.DOTALL
)

causes = []
for match in cause_pattern.finditer(content):
    cause_block = match.group(1)
    
    # Split individual causes, handle numbered and unnumbered items
    cause_items = re.findall(
        r'(?:^|\n)\s*(?:（?(\d+)[）.)]|(\d+)[\.。、]|)\s*([^\n]+)',
        cause_block
    )
    
    cleaned_causes = []
    for num1, num2, text in cause_items:
        if text.strip():
            cleaned_causes.append(text.strip().replace('\n', ' '))
    
    causes.append(cleaned_causes if cleaned_causes else [cause_block.strip().replace('\n', ' ')]) 

# --------------------------
# 5. Combine results
# --------------------------
result = [
    {"Quality Issue": issue, "Causes": cause}
    for issue, cause in zip(quality_issues, causes)
]

# --------------------------
# 6. Print results (for verification)
# --------------------------
for item in result:
    print(f"Quality Issue: {item['Quality Issue']}")
    print("Causes:")
    for i, cause in enumerate(item['Causes'], 1):
        print(f"  {i}. {cause}")
    print("---")

# --------------------------
# 7. Save causes to Excel
# --------------------------
all_causes = []
for item in result:
    all_causes.extend(item['Causes'])

df = pd.DataFrame({'Causes': all_causes})
output_path = "your_output_file.xlsx"  # Replace with your file path
df.to_excel(output_path, index=False, engine='openpyxl')
print(f"Data successfully saved to: {output_path}")
