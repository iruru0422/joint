import string
from spel.data_loader import dl_sa
import spel.configuration as conf
aida_mentions_vocab = conf.get_aida_plus_wikipedia_plus_out_of_domain_vocab()
entity = "1966_FIFA_World_Cup"
print(aida_mentions_vocab.get(entity, -1))