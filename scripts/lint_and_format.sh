#!/bin/bash

echo '*** PYLINT ***'
pylint *

# Excluding `data` excludes `datasets` since yapf uses fnmatch.fnmatch. Until
# a cleaner way to apply exclusions, run yapf on `datasets` separately after.
echo '*** YAPF ***'
yapf --recursive --in-place --parallel --verbose --exclude logs --exclude data .
yapf --recursive --in-place --parallel --verbose datasets/

echo '*** ISORT ***'
isort .
