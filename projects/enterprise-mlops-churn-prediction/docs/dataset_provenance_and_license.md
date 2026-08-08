# Dataset Provenance and License Record

## Dataset identity

| Item | Recorded value |
|---|---|
| Dataset | IBM Telco Customer Churn sample |
| Original distributed filename | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Repository filename | `data/raw/telco_customer_churn.csv` |
| Shape | 7,043 customer rows and 21 columns, including `Churn` |
| Local file size | 977,501 bytes |
| SHA-256 | `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a` |
| Verification date | 08 August 2026 |

The file is the widely used IBM Telco Customer Churn teaching sample. IBM's archived `customer-churn-prediction` code pattern identifies the source file by the original filename and links to the former Watson Analytics Community download. The Kaggle record maintained by BlastChar describes the same IBM sample and the same 7,043-customer schema.

## Provenance chain

1. **Original publisher/sample context:** IBM Telco Customer Churn sample data, originally distributed through the IBM Watson Analytics / Business Analytics community.
2. **IBM usage record:** [IBM archived customer-churn-prediction code pattern](https://github.com/IBM/customer-churn-prediction) identifies `WA_Fn-UseC_-Telco-Customer-Churn.csv` as its input dataset and preserves the historical source link.
3. **Public catalogue record:** [Kaggle - Telco Customer Churn, BlastChar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) describes the dataset as IBM sample data and records its data-file rights statement.
4. **Project copy:** the CSV was renamed to `telco_customer_churn.csv` for this repository. It is byte-identical to the earlier course-workspace copy in `ann_telcochurn_prediction_neuralnetworks/`, based on the SHA-256 value above. The original download transaction for the local copy was not preserved, so no more specific acquisition claim is made.

## Licensing and permitted-use statement

The BlastChar Kaggle dataset record labels the license as **“Data files © Original Authors.”** It does not identify CC0, Creative Commons, ODbL, or another standard open-data license.

IBM's archived code-pattern repository is licensed under Apache License 2.0, but its own license notice says that the Apache license covers the **code pattern** and that separately supplied third-party objects remain under their providers' separate licenses. The Apache-2.0 code license must therefore not be presented as a license for this CSV.

Accordingly:

- this project does not relicense the dataset;
- inclusion here is for academic analysis and reproducibility;
- copyright and other data rights remain with the original author(s) or rights holder(s);
- anyone redistributing the CSV or using it outside the academic submission should review the current IBM/Kaggle source terms and obtain any permission their use requires; and
- if redistribution is not permitted by the applicable submission or hosting rules, remove the CSV and require users to download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the cited source before running the pipeline.

## Reproduction and integrity check

After obtaining the source file, place it at `data/raw/telco_customer_churn.csv` and verify:

```bash
shasum -a 256 data/raw/telco_customer_churn.csv
```

Expected digest for the version evaluated in this project:

```text
88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a
```

The digest identifies the evaluated bytes; it does not itself establish ownership or grant a license.

## Suggested citation

> IBM, *Telco Customer Churn* sample dataset, originally distributed as `WA_Fn-UseC_-Telco-Customer-Churn.csv`; archived usage reference: IBM `customer-churn-prediction`; catalogue record: Kaggle `blastchar/telco-customer-churn` (accessed 08 August 2026).

## Source-access notes

- IBM repository status: archived/read-only when verified.
- Kaggle record rights label when verified: “Data files © Original Authors.”
- Access and verification date: 08 August 2026.
- This record intentionally distinguishes dataset rights from the licenses of code in either the IBM example or this project.
