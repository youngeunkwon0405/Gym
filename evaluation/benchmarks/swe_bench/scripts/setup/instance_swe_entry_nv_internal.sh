#!/usr/bin/env bash

source ~/.bashrc
SWEUTIL_DIR=/swe_util

# FIXME: Cannot read SWE_INSTANCE_ID from the environment variable
# SWE_INSTANCE_ID=django__django-11099
if [ -z "$SWE_INSTANCE_ID" ]; then
    echo "Error: SWE_INSTANCE_ID is not set." >&2
    exit 1
fi

# Read the swe-bench-test-lite.json file and extract the required item based on instance_id
item=$(jq --arg INSTANCE_ID "$SWE_INSTANCE_ID" '.[] | select(.instance_id == $INSTANCE_ID)' $SWEUTIL_DIR/eval_data/instances/swe-bench-instance.json)

if [[ -z "$item" ]]; then
  echo "No item found for the provided instance ID."
  exit 1
fi


WORKSPACE_NAME=$(echo "$item" | jq -r '(.repo | tostring) + "__" + (.version | tostring) | gsub("/"; "__")')

echo "WORKSPACE_NAME: $WORKSPACE_NAME"

# Clear the workspace
if [ -d /workspace ]; then
    rm -rf /workspace/*
else
    mkdir /workspace
fi
# Copy repo to workspace from /app (nv-internal-1 uses /app as source directory)
if [ -d /workspace/$WORKSPACE_NAME ]; then
    rm -rf /workspace/$WORKSPACE_NAME
fi
mkdir -p /workspace
cp -r /app /workspace/$WORKSPACE_NAME

# Note: nv-internal-1 supports Python, JavaScript, and TypeScript repos
# Environment activation is handled differently per language/project
# No conda activation is performed here as the container should already have the appropriate environment
