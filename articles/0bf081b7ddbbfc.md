---
title: "強化学習のマンカラ環境を作った話 - マルチエージェントRLライブラリ概観"
emoji: "♟️"
type: "tech"
topics:
  - "機械学習"
  - "強化学習"
  - "マンカラ"
  - "個人開発"
  - "python"
published: true
published_at: "2021-12-25 05:09"
---

# 初めに

この記事は[強化学習アドベントカレンダー 2021](https://adventar.org/calendars/6362)の記事として書かれたものです．
　初めまして，qqhann です．筑波大で修士をしており，修了の瀬戸際です．
　強化学習若手の会を知ったのは今年の初め頃だったと思います．Slack コミュニティに参加し，勉強会に参加してたまに質問させていただいたり，共有された記事を読んだりして，いつもためになっています．最近では，[ゼロから作る Deep Learning 4 のオープンレビュー](https://twitter.com/oreilly_japan/status/1466685904362553344?s=20)をそこで知り，通読させていただきました．レビューするつもりで文章を読むと集中力が違うからか，理解も進むように感じますね．強化学習若手の会にせっかく参加しているので，そこでもいつまでも読み専門というのも良くないなと思い，記事を書くことにしました．初めての Zenn 記事でもあります．

今年の前半に，強化学習を動かせるマンカラ環境を作成し，公開しました．
https://github.com/qqhann/Mancala
https://twitter.com/qqhann/status/1392126954028118019?s=21
当時は OpenSpiel と PettingZoo のコードを参考にして一気に実装しましたが，この記事を書くにあたって当時検討した API の特徴を再度調査してまとめました．少しでも有益な情報を提供できたらなと思います．なお，PettingZoo の論文を主に参照しました．
　続いて，作った時を振り返りながら，環境を作る時に得た知見や悩みポイントを書こうと思います．

# マルチエージェント学習の API

## Gym - POMDP

OpenAI/Gym はマルチエージェントの環境ではありませんが，強化学習におけるデファクトスタンダードのライブラリであり，どのライブラリもその設計思想に影響を受けていることから，まずおさらいします．Gym は次のサンプルコードで動きます．
[https://gym.openai.com/docs/](https://gym.openai.com/docs/)

```python
import gym

env = gym.make('CartPole-v0')

def policy(observation):
    return env.action_space.sample()

observation = env.reset()
for t in range(100):
    action = policy(observation)
    observation, reward, done, info = env.step(action)
    if done:
        break
env.close()
```

Gym の API は POMDP（partially observed Markov decision process）のパラダイムに則っています．環境から観察（observation）と報酬（reward）の情報が得られ，エージェントが選択した行動（action）を環境に伝えます．これがストレートにコードで表現されていて，読みやすく理解しやすいです．

## RLlib - POSG (partially observable stochastic games)

[https://docs.ray.io/en/releases-1.5.0/rllib-env.html#multi-agent-and-hierarchical](https://docs.ray.io/en/releases-1.5.0/rllib-env.html#multi-agent-and-hierarchical)

```python
from ray.rllib.env.multi_agent_env import make_multi_agent

env = make_multi_agent('CartPole-v0')

observation = env.reset()
for t in range(100):
    actions = policies(agents, observation)
    observation, rewards, dones, infos = env.step(acitons)
```

POSG のパラダイムでは，全てのエージェントは同時に観察し，報酬を得て，同時に行動を選択します．`observation`は，エージェント名をキーとした辞書になっていて，rewards, dones, acitons もこれに従います．厳密なターン制ゲーム以外では，この API は十分分かりやすく便利なものになっています．
　しかしこの API では，チェスやマンカラのような厳密なターン制ゲームでは，自分のターンではないエージェントは常にダミーの行動を選択するような工夫が必要になります．

## OpenSpiel - EFG (extensive form games)

[https://github.com/deepmind/open_spiel/blob/master/docs/concepts.md](https://github.com/deepmind/open_spiel/blob/master/docs/concepts.md)

```python
import pyspiel
import numpy as np

game = pyspiel.load_game("kuhn_poker")
state = game.new_initial_state()
while not state.is_terminal():
    if state.is_chance_node():
        # Sample a chance event outcome.
        action_list, prob_list = zip(*state.chance_outcomes())
        action = np.random.choice(action_list, p=prob_list)
        state.apply_action(action)
    else:
        # The algorithm can pick an action based on an observation (fully observable
        # games) or an information state (information available for that player)
        # We arbitrarily select the first available action as an example.
        action = state.legal_actions()[0]
        state.apply_action(action)
```

EFG は，ゲームを木として捉えます．全ての状態は木のノードであり，行動をすることで枝分かれします．確率的な要素がある場合は，「Chance（または Nature）プレイヤー」がいると仮定して，確率的な行動を実行させます．OpenSpiel は EFG のパラダイムに則ったライブラリです．ゲーム理論の実験でも使われているそうで，OpenSpiel の論文では例えば囚人のジレンマでの選択肢の変遷が分析例として図示されていました．
　いい点として，ゲームを木として捉えるため，探索による古典的アルゴリズムと相性がいいように思いました．一方，POSG や Gym の API と差異が大きく，比較するとコードが複雑に感じられやすいと思います．初学に不利になるほか，既存の強化学習アルゴリズムを試しにくくなります．

## PettingZoo - AEC (agent environment cycle) games

[https://www.pettingzoo.ml/api](https://www.pettingzoo.ml/api)

```python
from pettingzoo.butterfly import pistonball_v5

env = pistonball_v5.env()

env.reset()
for agent in env.agent_iter():
    observation, reward, done, info = env.last()
    action = policy(observation)
    env.step(action)
```

AEC のパラダイムは，エージェントが状態を観測し，行動を行い，他のエージェントからの報酬が与えられ，次に行動するエージェントが選ばれるという一連のことが，直列に行われます．PettingZoo は AEC を定義して作られたライブラリです．AEC のこの特徴によって，次のようなメリットがあります．

- エージェントの行動する順番の操作が自然に行える．
- API が Gym とかなり似ており，習得が容易．
- `last`関数によって，他のエージェントの行動を待たなくても reward にアクセスできる．

# マンカラ環境

## なぜ作ろうと思ったのか

マンカラとは，アフリカや中近東，東南アジアにかけて古くから遊ばれているボードゲームです．ローカルによるバージョン違いがありますが，[カラハ](https://en.wikipedia.org/wiki/Kalah)と呼ばれるルールが最も知られているかと思います．[世界のアソビ大全 51](https://www.nintendo.co.jp/switch/as7ta/games/)でこれを初めて知った日にどハマりし，6 時間くらい家族と遊んでいたように記憶しています．この頃強化学習も学び始めで，実装しながら学ぶいい機会だということで，マンカラの強化学習環境を作ることにしました．

## マンカラとは

![マンカラのボード](https://storage.googleapis.com/zenn-user-upload/e96e4fd0b11a-20211225.jpg)
（画像：[Wikipedia より](https://ja.wikipedia.org/wiki/%E3%83%9E%E3%83%B3%E3%82%AB%E3%83%A9)）
　マンカラは，二人向けのターン制ゲームです．それぞれのプレイヤーには 6 つのフィールドポケットと 1 つのポイントポケットがあります．フィールドポケットの中にはそれぞれ，ゲーム開始時に 4 つの石が入れられています．最初のプレイヤーは自分のフィールドポケットを一つ選択して石を全て手に取り，反時計回りに右のポケットに一つずつ石を落としていきます．自分のフィールドポケットの右側にあるポイントポケットに石を多く取り込んだ人の勝ちになります．
　ただし，自分のポイントポケットに石を落とした後も手に石が残っている場合，順番に相手のポケットにも石を落としていくことになります．石を落とし終えるとターン交代ですが，手に取った最後の石を自分のポイントポケットに落とした場合は相手のターンにならず続けてもう一度行動できます．また，自分の空になっているフィールドポケットに最後の石を落とした場合は，対岸の相手のフィールドポケットから石を総取りできます．これらのルールがマンカラを面白く駆け引きのあるゲームにしています．

## 設計

設計の際には次のことを意識しました．

- まず古典的なアルゴリズムにプレイ可能にする．
- 人間がプレイ可能にする．
- gym ライクな API を意識する．
- 強化学習のエージェントにプレイ可能にする．

このため，PettingZoo や OpenSpiel といったマルチエージェント強化学習のライブラリを参考にして，API 設計を行いました．出来上がったものはそれら二つの折衷案のような形になったのかなと思います．

```bash
pip install mancala
```

```python
from mancala.agents import HumanAgent, MiniMaxAgent
from mancala.mancala import MancalaEnv

player0 = HumanAgent(id=0)
player1 = MiniMaxAgent(id=1)
env = MancalaEnv(player0=player0, player1=player1)

state = env.reset()
done = False

while not done:
    if env.state.must_skip:
        print("Must skip")
    else:
        env.render()

    print("turn:", env.current_agent)
    act = env.current_agent.policy(state)
    state, reward, done = env.step(act)
print(f"Winner: {env.state._winner}")

```

おおよそこのようなコードで動くようになっています．
　`state`はベクトルではなくオブジェクトとなっており，エージェントがそれを考慮している必要があります．これは OpenSpiel と近いかもしれません．`env`が現在のエージェントを認識するようになっています．これは PettingZoo に近いかも．
　この環境に合わせて，古典的アルゴリズム（MiniMax 法，NegaScout）のエージェントや，ニューラルネットのエージェント（A3C）を実装してあります．

## 工夫・悩みポイント

### スキップ

マンカラにはスキップというルールがあります．最後の石がポイントポケットに入ったら相手のターンはスキップされます．悩んだ結果，最終的には「スキップしなければならない」という情報を状態に持たせて，エージェントにスキップという行動を取らせることにしました．これ以外の方法では，エージェントのとりうる行動が可変長になってしまうと思われて，設計ができそうになかったためです．

### ボードをひっくり返すか

ターンが切り替わったらボードをひっくり返すべきかどうか，結構悩みました．実装的には，ボードのベクトルをコピーして，前半後半を逆にすることです．最終的にはひっくり返したような記憶がありますが，かなり抵抗しました．というのも，先手後手で先にプレイしたという情報に差があるのだから，それを判別可能にすべきだと思っていました．

### 可能な行動（legal action）以外の扱い

空のポケットには石がないので，そこから石を取る選択をすることはできません．ニューラルネットのエージェントに学習させる時，このルール上選択できない手（illegal move）をとってしまうことがあります．当時の実装では illegal move ではペナルティを与えてゲーム終了することにしました．しかし，学習効率を考えると illegal move を教えるためだけにゲームを終了させていることになるので，今後はそもそも選択できないようにしようと思っています．しかし，行動空間が可変長になるようで，どうも引っ掛かるんですよね．実装上は，illegal move の選択確率を 0 でマスクかければいいのですけど．

### ポイントポケットの石数を含めるか

ニューラルネットのエージェントを学習させる時，ボードの情報としてポイントポケットの石の数まで含めるかどうか，悩みました．当時の実装では含めています．現在の獲得ずみ石数が過半数に届きそうであれば大胆な行動を選択しても良くなり，その情報としてあった方がいいと考えたためです．しかし実際には学習のノイズとなっている可能性があるため，今変更を行うとしたら外して試そうかなと思っています．

## 実装してみての発見

### 一番遅いのは…

実験用のコードとして，ニューラルネットのエージェント（A3C）を訓練させた後，実装ずみのアルゴリズムで総当たり戦ができるようにしました．NegaScout は先の手を読んで行動を選択するアルゴリズムですが，読みの深さを 2 にすると A3C と互角くらいなのですが，深さ 4 にすると一気に遅くなります．現状の自分のコードでは A3C を深さ 2 の NegaScout には勝てるくらいにできますが，深さ 4 には勝てません．しかし学習をうまくして勝てるようになれば，探索アルゴリズムのように指数関数的に処理時間が増えないはずですから，やりがいがあるなと思います．そんなわけで一番遅いエージェントはというと…，NegaScout 探索を無駄に深くしない限り，人間が最も遅いです．HumanAgent というクラス名を見ると人間もまたエージェントの一つなのだとしみじみした気持ちになります．

また，全てピュア Python で実装したのですが，PyTorch のモデルを CUDA で訓練させてもあまり高速化しませんでした．これにはやはり環境と古典的アルゴリズムも Python で実装していることが原因なのかなと思います．実際，他のライブラリを見るとゲームのコア部分を C++で実装しているものが多かったので，そこから高速化は始まっているんだなーという感想を持ちました．

# 終わりに

オチとして，マンカラは二人ゼロ和完全情報ゲームに分類されるのですが，解析がかなりなされていて，カラハのルールにおいては先手必勝が結論出されていました．実装を終えてからこれを知り，強化学習でやる意味はあったのだろうかと気を落としましたが，とはいえ，実装したことに悔いはありません（笑）．

当時表面上のコードを真似するだけでしたが，この記事を書くにあたって，それぞれの API の特徴と設計思想を深く調べるきっかけとなりました．修士論文のために古典的な強化学習のアルゴリズムからオフライン強化学習や階層型強化学習を実装してみた今改めて考えると，共通の API って偉大だなと思います．もし今作り直すとしたら PettingZoo の API をそのまま採用するかもしれません．この方が公開されている既存の実装をそのまま試せそうだからです．

最後まで読んでいただきありがとうございました．GitHub でスターをつけてもらえるとアップデートしたり新しいリポジトリを生やすモチベーションにあるので，いいなと思ったかはぜひ[GitHub](https://github.com/qqhann/Mancala)も見ていってください．

それでは，良い年末・良いお年を．

# 参照文献

- PettingZoo
  - Terry, Justin K., et al. "Pettingzoo: Gym for multi-agent reinforcement learning." arXiv preprint arXiv:2009.14471 (2020). [https://arxiv.org/abs/2009.14471](https://arxiv.org/abs/2009.14471)
  - GitHub: [https://github.com/Farama-Foundation/PettingZoo](https://github.com/Farama-Foundation/PettingZoo)
- OpenSpiel
  - Lanctot, Marc, et al. "OpenSpiel: A framework for reinforcement learning in games." arXiv preprint arXiv:1908.09453 (2019). [https://arxiv.org/abs/1908.09453](https://arxiv.org/abs/1908.09453)
  - GitHub: [https://github.com/deepmind/open_spiel](https://github.com/deepmind/open_spiel)
- RLlib
  - GitHub: [https://github.com/ray-project/ray](https://github.com/ray-project/ray)
