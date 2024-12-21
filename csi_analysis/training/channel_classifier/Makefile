#################################################################################
# GLOBALS                                                                       #
#################################################################################
PYTHON	=	python3.12
PROJECT_DIR	=	$(CURDIR)
#################################################################################
# COMMANDS                                                                      #
#################################################################################
# Makefile
POETRY_INSTALL_URL=https://install.python-poetry.org

# Define the virtual environment directory
VENV_DIR=.venv

# Install Poetry
.PHONY:	install-poetry
install-poetry:
	curl	-sSL	$(POETRY_INSTALL_URL)	|	python3	-

# Install production dependencies
.PHONY:	install
install:	install-poetry
	poetry	install

# Install development dependencies
.PHONY:	install-dev
install-dev:	install
	poetry	install	--dev

# Update the dependencies
.PHONY:	update
update:
	poetry	update

# Create data
.PHONY:	create-data
create-data:
	poetry	run	python channel_classifier/dataset.py

# Delete data except immutable raw data
.PHONY:	delete-data
clean-data:
	rm	-rf	./data/processed/*
	rm	-rf	./data/external/*
	rm	-rf	./data/interim/*

# Train the model
.PHONY:	train
train:
	poetry	run	python channel_classifier/modeling/train.py

# Clean the project
.PHONY:	clean
clean:
	rm	-rf	$(VENV_DIR)
	find	.	-type	d	-name	"__pycache__"	-exec	rm	-r	{}	+
	find	.	-type	f	-name	"*.pyc"	-exec	rm	-r	{}	+