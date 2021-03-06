# Sentence Similarity Checker using Encoder Decoder Model
by Makoto Morishita
(morishita.makoto.mb1 [[at]] is.naist.jp)

## これは何?
ニューラル機械翻訳 (NMT) のモデルとして有名な
Encoder Decoder モデルを使った文の類似度を計算するツールです．
論文でいうと Sutskever et. al. 2014 (https://arxiv.org/abs/1409.3215) を応用しています． 論文中でも述べられていますが， Encoder が出力するベクトルには，
同じような位置に同じような意味のベクトルが集まる特徴があり，
これを利用することで，意味的に近い文を見つけます．

## 必要なもの
* chainer (http://chainer.org)
* KyTea (http://www.phontron.com/kytea/index-ja.html)
* パラレルコーパス (対訳コーパス，モデルを学習する場合，今回は日英を仮定．)

## 事前準備
chainer と KyTea を使えるようにしておいてください．

## 使い方
このプログラムは大きく分けて，「モデル学習」と「類似度計算」の2つに大きく分けられます．

### モデル学習
モデル学習では，通常通り Encoder Decoder モデルを学習します．  
その後，できたモデルを使って類似度の比較対象になる文ベクトルを計算しておきます．  
サンプルモデルを利用する場合は下記の手順を，  
学習から自分で行う場合はパラレルコーパスを単語分割する部分から読んで下さい．  

#### サンプルモデルをダウンロードする
シェルスクリプトを実行するだけでダウンロードできるはずです．
```
./download_sample.sh
```

sample ディレクトリができます．
この中に，サンプルモデルとサンプルトレーニングデータの文ベクトルがあります．

下記にモデルの学習方法を述べますが，もし類似度計算を試してみるだけであれば  
付属のサンプルモデルを使うだけで出来ます．  
その場合は下記の手順は飛ばして類似度計算まで進んでください．  

#### パラレルコーパスを単語分割する
英語はMoses tokenizer (https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl) を使って単語分割してください．
詳細は割愛  

日本語はKyTeaを使って単語分割する  
```
/path/to/kytea -model /path/to/model.bin -notags -wsconst D < /path/to/train.ja > /path/to/train.tok.ja  
/path/to/kytea -model /path/to/model.bin -notags -wsconst D < /path/to/test.ja > /path/to/test.tok.ja  
```

#### Vocabulary を作成する
NMTでは使用する語彙の数を制限する必要があります．  
今回はコーパス内の頻度順に語彙を作成して，低頻度の語については未知語として処理します．  
また，今回は語彙数を1万語とします．  

```
python3 make_vocab.py --input /path/to/train.tok.ja --output /path/to/vocab.ja --size 10000  
python3 make_vocab.py --input /path/to/train.tok.en --output /path/to/vocab.en --size 10000  
```

#### ID化したコーパスを作成する
先程作成した Vocabulary を基に，ID化したコーパスファイルを作成します．  
train と test の両方を処理します．  

```
python3 apply_vocab.py --input /path/to/train.ja --output /path/to/train.id.ja --vocab /path/to/vocab.ja  
python3 apply_vocab.py --input /path/to/train.en --output /path/to/train.id.en --vocab /path/to/vocab.en  
python3 apply_vocab.py --input /path/to/test.ja --output /path/to/test.id.ja --vocab /path/to/vocab.ja  
python3 apply_vocab.py --input /path/to/test.ja --output /path/to/test.id.en --vocab /path/to/vocab.en  
```

#### Encoder Decoder モデル学習
実際にモデルを学習する．  

```
python3 train.py --source /path/to/train.id.ja --target /path/to/train.id.en --source_test /path/to/test.id.ja --target_test /path/to/test.id.en --srcvocab 10000 --trgvocab 10000 --embed 1024 --hidden 1024 --batchsize 128 --test_batchsize 1 --out ./result --gpu 0  
```

result ディレクトリ内に学習したモデルと，テストセットを翻訳したものができる．  

もし必要ならBLEUを測る．(1の部分はepoch数なので，2, 3と変更して良い)  
```
python3 bleu.py --ref /path/to/test.id.en --hyp result/1  
```

#### トレーニングセットのベクトルを計算する
比較対象とする文のベクトルを計算する．  

```
python3 vector_representation.py --model_file /path/to/result/model_epoch_30 --source /path/to/train.id.ja --srcvocab 10000 --trgvocab 10000 --embed 1024 --hidden 1024 > /path/to/train.vec.ja  
```

### 類似度計算
学習したモデルと事前に計算した文ベクトルを使って，  
与えられたテスト文に意味的に近い文を見つけます．  

```
echo "何か飲み物をください" | /path/to/kytea -model /path/to/model.bin -notags -wsconst D | python apply_vocab.py --input /dev/stdin --output /dev/stdout --vocab /path/to/vocab.ja | python check_similar_sentence.py --vector /path/to/train.vec.ja --sentence /path/to/train.ja --model_file /path/to/result/model_epoch_30  
```

テスト文に近い文が表示される．  
左に表示されるのはスコア (文ベクトルとのコサイン類似度)．  

    0.22045767155   何 か 暖か い 飲み物 を くださ い 。
    0.295753118189  何 か 飲物 を くださ い 。
    0.30510591702   何 か これ を 切 る もの を 貸 し て くださ い 。
    0.318465333711  何 か 書 く もの を 貸 し て くれ 。
    0.329296079882  何 か 飲み物 を いただけ ま す か 。
    0.330015366557  何 か 飲みもの を くださ い 。
    0.332928890929  何 か 熱 い 飲み物 を 下さ い 。
    0.371530655737  何 か 食べ物 を 下さ い 。
    0.406141231431  何 か 飲 む もの を くれ 。
    0.411517417135  何 か 冷た い 飲み物 を 下さ い 。

#### サンプルファイルを使う場合
もしサンプルを動かす場合こうなる．  

```
echo "何か飲み物をください" | /path/to/kytea -model /path/to/model.bin -notags -wsconst D | python apply_vocab.py --input /dev/stdin --output /dev/stdout --vocab ./data/vocab/vocab.10000.ja | python check_similar_sentence.py --vector ./sample/train.vec.ja --sentence ./data/raw/train.ja --model_file ./sample/sample_model  
```
