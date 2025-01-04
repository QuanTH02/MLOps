import sys
import os

for path in sys.path:
    print(path)

print('##################')
print(os.path.abspath(os.path.dirname(__file__)))
print('##################')

sys.path.append(r'C:\Code\Project_MLops\data\crawl')

for path in sys.path:
    print(path)