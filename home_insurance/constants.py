DATES_COLS = ["QUOTE_DATE", "COVER_START", "P1_DOB", "MTA_DATE"]

BINARY_COLS = ['CLAIM3YEARS', 'BUS_USE', 'CLERICAL', 'AD_BUILDINGS', 'AD_CONTENTS', 'CONTENTS_COVER', 'BUILDINGS_COVER',
               'P1_POLICY_REFUSED', 'APPR_ALARM', 'APPR_LOCKS', 'FLOODING', 'NEIGH_WATCH', 'SAFE_INSTALLED',
               'SEC_DISC_REQ', 'SUBSIDENCE', 'LEGAL_ADDON_PRE_REN', 'LEGAL_ADDON_POST_REN', 'HOME_EM_ADDON_PRE_REN',
               'HOME_EM_ADDON_POST_REN', 'GARDEN_ADDON_PRE_REN', 'GARDEN_ADDON_POST_REN', 'KEYCARE_ADDON_PRE_REN',
               'KEYCARE_ADDON_POST_REN', 'HP1_ADDON_PRE_REN', 'HP1_ADDON_POST_REN', 'HP2_ADDON_PRE_REN',
               'HP2_ADDON_POST_REN', 'HP3_ADDON_PRE_REN', 'HP3_ADDON_POST_REN', 'MTA_FLAG']

MULTI_CAT_COLS = [
    'CLAIM3YEARS',
    'P1_EMP_STATUS',
    'P1_PT_EMP_STATUS',
    'BUS_USE',
    'CLERICAL',
    'AD_BUILDINGS',
    'AD_CONTENTS',
    'CONTENTS_COVER',
    'BUILDINGS_COVER',
    'P1_MAR_STATUS',
    'P1_POLICY_REFUSED',
    'P1_SEX',
    'APPR_ALARM',
    'APPR_LOCKS',
    'FLOODING',
    'NEIGH_WATCH',
    'OCC_STATUS',
    'SAFE_INSTALLED',
    'SEC_DISC_REQ',
    'SUBSIDENCE',
    'PAYMENT_METHOD',
    'LEGAL_ADDON_PRE_REN',
    'LEGAL_ADDON_POST_REN',
    'HOME_EM_ADDON_PRE_REN',
    'HOME_EM_ADDON_POST_REN',
    'GARDEN_ADDON_PRE_REN',
    'GARDEN_ADDON_POST_REN',
    'KEYCARE_ADDON_PRE_REN',
    'KEYCARE_ADDON_POST_REN',
    'HP1_ADDON_PRE_REN',
    'HP1_ADDON_POST_REN',
    'HP2_ADDON_PRE_REN',
    'HP2_ADDON_POST_REN',
    'HP3_ADDON_PRE_REN',
    'HP3_ADDON_POST_REN',
    'MTA_FLAG']

CATEGORICAL_FEATURES = MULTI_CAT_COLS
