## Kokoro tokenizer — single-pass UTF-8 character tokenizer.
## Each IPA phoneme symbol is one token. Vocabulary loaded from GGUF metadata.

import std/tables
import ggml/gguf_loader

type
  Tokenizer* = object
    tokens*: seq[string]
    tokenMap*: Table[string, uint32]
    eosId*: uint32
    padId*: uint32

proc loadTokenizer*(model: GgufModel): Tokenizer =
  ## Load tokenizer from GGUF metadata arrays.
  let ggufCtx = model.ggufCtx
  let tokensKeyId = gguf_find_key(ggufCtx, "tokenizer.ggml.tokens")
  if tokensKeyId < 0:
    raise newException(KeyError, "tokenizer.ggml.tokens not found in GGUF")

  let nTokens = gguf_get_arr_n(ggufCtx, tokensKeyId)
  var tokenizer = Tokenizer(
    eosId: model.getU32("tokenizer.ggml.eos_token_id"),
    padId: model.getU32("tokenizer.ggml.padding_token_id"),
  )

  for i in 0..<nTokens:
    let tok = $gguf_get_arr_str(ggufCtx, tokensKeyId, i)
    tokenizer.tokens.add(tok)
    tokenizer.tokenMap[tok] = uint32(i)

  return tokenizer

proc tokenize*(t: Tokenizer, text: string): seq[uint32] =
  ## Tokenize a phonemized string into token IDs.
  ## Kokoro uses single UTF-8 character tokens.
  var i = 0
  while i < text.len:
    # Determine UTF-8 character length
    let b = uint8(text[i])
    let charLen = if b < 0x80: 1
                  elif b < 0xE0: 2
                  elif b < 0xF0: 3
                  else: 4
    let ch = text[i..<min(i + charLen, text.len)]
    if t.tokenMap.hasKey(ch):
      result.add(t.tokenMap[ch])
    # Skip unknown characters silently
    i += charLen

proc wrapTokens*(t: Tokenizer, tokens: seq[uint32], bosId: uint32 = 0): seq[uint32] =
  ## Wrap token sequence with BOS and EOS tokens.
  result.add(bosId)
  result.add(tokens)
  result.add(t.eosId)
