from spel.data_loader import dl_sa

mention_etoa = dl_sa.aida_canonical_redirects
mention_atoe = dict()
for key, value in mention_etoa.items():
    mention_atoe[value] = key
print(mention_atoe)