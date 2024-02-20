#!/bin/bash

# Read the CSV file and iterate over each line
teamID=""
while IFS=',' read -r ID; do
    # Process only non-header lines
    if [[ "$ID" != "ID" ]]; then
        if ! [[ -n "$teamID" ]]; then
            teamID="$ID"
        else
            if [[ -n "$ID" ]]; then
                echo "$ID"
                setfacl -Rm "u:$ID:rwX,d:u:$ID:rwX" "/hkfs/work/workspace_haic/scratch/qx6387-$teamID/"
            fi
        fi

    fi
done < team.csv
