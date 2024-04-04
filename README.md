# Mini-Machine-Translation: Learning about Cross-Attention in Transformers

- before the project, i only knew about the decoder part of the transformer (I mainly use transformers for pure autoregressive modelling)
- cross-attention is super important for conditioning the transformer on some text (like the text prompt in DALLE-1)
- we use cross attetion to learn to translate from English to German
- the data set is taken from: https://www.kaggle.com/datasets/kaushal2896/english-to-german

Things I learned:

- the cross attention matrix does not have to be square (it is non-square when (padded) decoder and encoder sequence have different lengths)
- the cross attention matrix is not masked (since we use keys and values from the encoder sequence, we never attend to decoder tokens; we can't possibly attend to future tokens)
- PyTorchs forward function can handle multiple arguments (like decoder and encoder sequence) but nn.Sequential doesn't work anymore -> use nn.ModuleList
- the target sequence (in our case german tokens) are passed to the transformer but we have to shift them to the right by one and add a start token.
  otherwise, the transformer can learn the identity function with the decoder and ignore the encoder output because the targets are leaking into the data instances
- translating from English (with English tokens) to German (with German tokens) (both sets of tokens can be different) is analogous to translating text tokens (from the DALLE user) to image tokens that are used by a VQVAE

## Training

| Transformer Hyperparameters | Value |
|-----------------------------|-------|
| d_model                     | 256   |
| n_heads                     | 16     |
| n_layers (enc, dec)         | 4, 4    |
| n_symbols (enc, dec)        | 95, 120 |
| context_size                | 128   |
| p_dropout                   | 0.1   |

- trained on 200k sentence pairs (english - german)
- learning rate = 3e-4 (for first two episodes, then 1e-4 for the rest of the training)
- Adam optimizer
- batch size = 128
- epochs = 10 (training takes approx. 20 minutes on an RTX3060)

## Results

| Hyperparameters for Sampling  |          Details          |
|-------------------------------|---------------------------|
| Sequence Generation           | 8 sequences at the same time |
| Transformer Temperature       | 0.7                      |
| n_tokens                      |  60                     |

- Inference time (8 generations): approx. 1s (on an RTX3060)

----

Source: Hi, how are you?

Translation:
1. Hast du dein Hund ihr gesagt?
2. Wer ist dein Haar heiße? Haus Haus?
3. Wie haben ihr Ihre Wage, weine dann haben macht?
4. Hast du dir Ihre Aust gesagt, Tom weg?
5. Hast du dir eine Austüto?
6. Warum ist deine Ihre Städter, dass du gewacht hast?
7. Warum ist dir Ihr, dass du heute gehalten hast?
8. Wer ist dein habe, dein Haarte?

----

Source: What's your name?

Translation:
1. Wer ist dein Australien?
2. Welcher Getantstertelligen Sie wein dein Sind?
3. Welche Nache mein Haus?
4. Was ist deine Karzawatten?
5. Wo hast du deine Zeitname betrinken?
6. Welche Mauchen lernen Sie nicht deine Gerungsten?
7. Wer ist dein Leblingster?“ „Sie gibt du?
8. Welchen Sie heißt deine Tage?

----

Source: My name is Damian.

Translation:
1. Meine Familie ist eine Schwariftaminische Coxa.
2. Mein Meinaa ist in Mannana.
3. Meine Manainat ist eine gute Manny in Coma Manieen.
4. Meine Leute, der Mann ist jemand in Gei Cregns.
5. Mein Kanaer ist eine spielen Manna.
6. Mein Mann ist ein Fenstamilieri.
7. Mein Mana ist eine mehr Crana.
8. Man hat eine Manninanamiens eine wichtiger Wäliesta.

---

Source: That's an unusual name!

Translation:
1. Das ist ein nur eine eine und so denn?
2. Das ist ein so nur wahr, das ein Spaß anschienereit.
3. Das ist einfach ein für eine sondere Erfahren.
4. Das ist eine sonderker Spadtzier!
5. Das ist ein und eine sonder so so guter Eltern.
6. Das ist eine von Kanadiere Taibalien.
7. Danaus ist ein In, und eine sehr bendessund Sie nur gehenaus
8. Das hat ein so Jannisches mehr und nur Sonnund.

---

Source: You think? What's your name?

Translation:
1. Du glaubst weiß, dass das und gefällt hast Tom?
2. Ihr glauben nicht, dass du doch glauben würden wirklich?
3. Du denkst geschäftigt, dass du denken Familie hast, was?
4. Du glaubst den Freund, dass ihr ihr wohl?
5. Du glaubst den Entschlag, Tom kennst süchter.
6. Du glauben sie den Mann, dass Tom seine Fraue, wahr?
7. Du glauben den Minumm, dass das ich bist haben wir hassen?
8. Du hast deine Frau, wenn ihr?

---

Source: My name is Tom and I can speak German.

Translation:
1. Mein Kann, dass ich Tom ist einen Elter spricht.
2. Ich glaube mich, dass Tom Maria kennen sind ein Baum.
3. Meine Mann rennen meine Ahnung schlafen, dass Tom ist.
4. Ich und Maria war mein pinigen Vorsamenam.
5. Meine Freunde ist weiß, der ich viele Französisch sprechen b
6. Als Maria kann mein Frauen hat von Tom.
7. Meine Mann und ich für wie Tom nach Jokanne.
8. Meine Mann, dass ich Tom kennen beide Maria in Johannes klar

--- 

Source: Your German is horrible!

Translation:
1. Dein Mutter ist aus in der erfür neunklicherlos in Aliere.
2. Dein Bruder ist neuer werde selts schwer gegen an.
3. Die Mann ist genöt beim von der Zuzsobidbenibtinnen ausmas z
4. Dein Mil ist dieser lauten für der ernsten lauosterlericht.
5. Dein Bruder ist viegen alle meine Großsonne.
6. Haben Sie deine Mahrben in seiner Mädchchen im Leben halt.
7. Deine Schume ist schwer welbster Bruderren.
8. Dein Leben ist auf der Büchstüngstlicher.

---

Source: Bruh, I am using a transformer that has been trained for 20 minutes.

Translation:
1. In der Augung, leid ist nicht mehr andere ausstellen in well
2. Man ist von die Auto in so und noch danausantes sich so geba
3. Einen Sommer, dass er im Gendschuhlich nicht beschäft hinder
4. Der Mutter ist mir der nur der Morgen laufen auf den Aungebe
5. Ein Sold schnell als ich er seine Pinon auf.
6. Ja gibt so ein Mennern meinen Minutionschen und in der Staue
7. Mit den Mutter ist nächste Branz in der Schafen in der Beste
8. Man ist ein seltster Miuraum nicht der Sonnee er mit meinem

---

NOTE:

- all this Tom and Maria stuff that the transformer outputs is due to the training data.
  Apparently, all male names have been replaced by Tom and all female names by Maria.

- the transformer is pretty trash at attending to the input.
  To improve the model, we could:
    - increase the model capacity (d_model, more heads, more layers)
    - add a proper tokenizer (currently one token = one character)
    - increase the dataset (the short phrases are not sufficient for real translation a la DeepL)
    - train for longer
