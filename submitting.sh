#!/usr/bin/env bash

FILEDEST="submit/submission.csv"
TIMESTAMP=$(date +%s)
SAVETO="all_submissions/submission_at_$TIMESTAMP.csv"

head -n 10 ${FILEDEST}
echo "Submitting file: $FILEDEST. Current timestamp: $TIMESTAMP"

cp ${FILEDEST} $SAVETO
echo "$FILEDEST was copied to $SAVETO"

kaggle competitions submit -c ieee-fraud-detection -f ${FILEDEST} -m "Submit at $TIMESTAMP"
