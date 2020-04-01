from clswgan import CLSWGANGP
wgan = CLSWGANGP()
wgan.train(epochs=30, batch_size=64, sample_interval=10)
