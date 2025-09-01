#!/bin/bash
poetry run uwsgi --master --workers 3 --protocol http --socket 0.0.0.0:5001 --module ml_app:app
