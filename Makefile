.PHONY: setup pipeline train backtest forecast app test clean all

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

pipeline:
	python src/data/collect_headlines.py
	python src/data/fetch_prices.py
	python src/features/sentiment_scorer.py
	python src/features/index_builder.py

train:
	python src/models/train.py

backtest:
	python src/models/backtest.py

forecast:
	python src/models/forecast.py

all: pipeline train backtest forecast

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v --tb=short

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
