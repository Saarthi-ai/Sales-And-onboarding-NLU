import json
import gspread
import pandas as pd


def filter_tags(tags_str):
    tags = tags_str.split()
    filtered_tags = []

    for tag in tags:
        # Extract the entity type from the tag, e.g., "U-date" -> "date"
        entity_type = tag.split('-')[-1] if '-' in tag else None

        if entity_type and entity_type in entity_types:
            filtered_tags.append(tag)
        else:
            filtered_tags.append("O")

    return ' '.join(filtered_tags)

tagging_scheme = 'BILOU'
ner_sheet_name = f'[BFSI] {{{tagging_scheme} | NER}} Training data'
gspread_client = gspread.service_account(filename='./inputs/credentials.json')

sheet = gspread_client.open(ner_sheet_name)
languages = ['english', 'hindi']
entity_types = ['ptp_date', 'time']

dfs = []
for language in languages:
    lang_df = pd.DataFrame(sheet.worksheet(language).get_all_records()).dropna()
    entity_types_df = lang_df[lang_df['type'].apply(lambda x: any(substring in x for substring in entity_types))].drop_duplicates(subset=['text']).reset_index(drop=True)
    dfs.append(entity_types_df)

final_df = pd.concat(dfs)
final_df['ner'] = final_df['tags'].apply(filter_tags)
print(f'Length of final df: {len(final_df)}')
del final_df['tags']
del final_df['type']

ner_labels = [f'{scheme_tag}-{entity_type}' for entity_type in entity_types for scheme_tag in tagging_scheme if scheme_tag != 'O']
ner_labels.extend(['O', '<pad>'])

final_df.to_csv('./inputs/ner/data.csv', index=False)
with open('./inputs/ner/labels.json', 'w') as f:
    json.dump({'ner': ner_labels}, f, indent=2)
