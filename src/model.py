"""Encoder-Decoder (seq2seq) GRU forecaster with optional additive attention."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


class Seq2Seq(keras.Model):
    """Encoder: GRU reads the L past steps -> outputs h_1..h_L and final state.
    Decoder: GRU cell runs H steps. At step k its input is
        [previous humidity (true during teacher forcing, own prediction otherwise),
         known future calendar features, attention context vector].
    Attention (Bahdanau / additive): score_i = v^T tanh(W1 h_i + W2 s_{k-1});
    alpha = softmax(score); context = sum_i alpha_i h_i.
    Without attention the context is the encoder's last hidden state (the
    fixed-length context vector c of the lecture)."""

    def __init__(self, units=64, horizon=5, use_attention=True, dropout=0.2, **kw):
        super().__init__(**kw)
        self.units, self.horizon, self.use_attention = units, horizon, use_attention
        self.enc = layers.GRU(units, return_sequences=True, return_state=True)
        self.drop = layers.Dropout(dropout)
        self.cell = layers.GRUCell(units)
        self.W1 = layers.Dense(units, use_bias=False)
        self.W2 = layers.Dense(units, use_bias=False)
        self.v = layers.Dense(1, use_bias=False)
        self.fc = layers.Dense(units, activation="relu")
        self.out = layers.Dense(1)

    def _context(self, enc_out, enc_last, s):
        if not self.use_attention:
            return enc_last, None
        score = self.v(tf.nn.tanh(self.W1(enc_out) + self.W2(s)[:, None, :]))   # (B,L,1)
        alpha = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(alpha * enc_out, axis=1), alpha[..., 0]

    def call(self, inputs, training=False, return_attention=False):
        """inputs = (X, DC, y_last) for free-running decoding, or
        (X, DC, y_last, Y_true) for teacher forcing."""
        X, DC, y_last = inputs[:3]
        Y_true = inputs[3] if len(inputs) > 3 else None
        enc_out, s = self.enc(X, training=training)
        enc_out = self.drop(enc_out, training=training)
        enc_last = s
        prev, preds, atts = y_last[:, None], [], []
        for k in range(self.horizon):
            ctx, a = self._context(enc_out, enc_last, s)
            x_t = tf.concat([prev, DC[:, k], ctx], axis=-1)
            s, _ = self.cell(x_t, [s], training=training)
            h = self.fc(tf.concat([s, ctx], axis=-1))
            y_hat = self.out(self.drop(h, training=training))
            preds.append(y_hat)
            if a is not None:
                atts.append(a)
            prev = Y_true[:, k:k + 1] if Y_true is not None else y_hat
        yp = tf.concat(preds, axis=1)
        if return_attention:
            return yp, (tf.stack(atts, axis=1) if atts else None)
        return yp


def _batches(d, bs, shuffle, rng=None):
    n = len(d["X"]); idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield d["X"][j], d["DC"][j], d["YL"][j], d["Y"][j]


def fit(data, cfg, use_attention=True, verbose=True):
    mc = cfg["model"]
    tf.keras.utils.set_random_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = Seq2Seq(mc["units"], cfg["horizon"], use_attention, mc["dropout"])
    opt = keras.optimizers.Adam(mc["learning_rate"], clipnorm=mc["clipnorm"])
    loss_fn = keras.losses.MeanSquaredError()

    @tf.function(reduce_retracing=True)
    def step_tf(x, dc, yl, y):
        with tf.GradientTape() as tape:
            loss = loss_fn(y, model((x, dc, yl, y), training=True))
        opt.apply(tape.gradient(loss, model.trainable_variables), model.trainable_variables)
        return loss

    @tf.function(reduce_retracing=True)
    def step_free(x, dc, yl, y):
        with tf.GradientTape() as tape:
            loss = loss_fn(y, model((x, dc, yl), training=True))
        opt.apply(tape.gradient(loss, model.trainable_variables), model.trainable_variables)
        return loss

    best, best_w, wait, hist = np.inf, None, 0, []
    for ep in range(mc["max_epochs"]):
        tl = []
        for x, dc, yl, y in _batches(data["train"], mc["batch_size"], True, rng):
            use_tf = rng.random() < mc["teacher_forcing_prob"]
            tl.append(float((step_tf if use_tf else step_free)(x, dc, yl, y)))
        vp = predict(model, data["val"], mc["batch_size"])          # free-running, like at test time
        vl = float(np.mean((vp - data["val"]["Y"]) ** 2))
        hist.append((np.mean(tl), vl))
        if verbose:
            print(f"epoch {ep+1:02d}  train_mse={np.mean(tl):.4f}  val_mse={vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, best_w, wait = vl, model.get_weights(), 0
        else:
            wait += 1
            if wait >= mc["patience"]:
                break
    model.set_weights(best_w)
    return model, hist


def predict(model, d, bs=512):
    out = [model((d["X"][i:i + bs], d["DC"][i:i + bs], d["YL"][i:i + bs]), training=False).numpy()
           for i in range(0, len(d["X"]), bs)]
    return np.concatenate(out)


def attention_weights(model, d, n=256):
    _, a = model((d["X"][:n], d["DC"][:n], d["YL"][:n]), training=False, return_attention=True)
    return None if a is None else a.numpy()
