rm dist/
python3 setup.py sdist
twine upload --verbose dist/*
