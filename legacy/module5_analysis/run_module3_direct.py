from pipeline.extract import extract
from pipeline.clean import clean
from pipeline.anonymize import anonymize
from pipeline.features import engineer_features
from pipeline.bias_check import run_bias_checks
from pipeline.lineage import write_outputs
df=extract(); n=len(df)
df,cr=clean(df); df,ar=anonymize(df); df,eng=engineer_features(df); b=run_bias_checks(df)
q={"extract_rows":n,"validation_passed":None,"clean_rows_out":cr["rows_out"],"anonymize_rows_out":len(df),"cleaning":cr}
write_outputs(df,q,b,ar)
print(cr); import json; print(json.dumps(b,indent=1,default=str)); print(df.shape, df.columns.tolist())
