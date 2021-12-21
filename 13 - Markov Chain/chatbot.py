txt = "I am a good boy I am singer and I do dancing. Hey this is jayant bhai"
import markovify

model = markovify.NewlineText(txt, state_size =2)
print(model.make_sentence())