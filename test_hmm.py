from hmmlearn.hmm import GMMHMM
model = GMMHMM(n_components=3)
print([m for m in dir(model) if not m.startswith('__')])
