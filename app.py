import streamlit as st
import librosa
import librosa.display
import numpy as np
from scipy import signal
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import base64
import parselmouth
from parselmouth.praat import call

HOP = 1000
GRAPH_WIDTH = 1200
GRAPH_HEIGHT = 400


@st.cache
def calc_fo(wav):
    fo, voiced_flag, voiced_prob = librosa.pyin(wav, 80, 2000)
    return fo


@st.cache
def measurePitch(wav):
    sound = parselmouth.Sound(wav)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    return hnr


@st.cache
def calc_spectrum(wav, sr, fo):
    spectrum = np.abs(np.fft.fft(wav, sr)[:int(sr / 2)])
    freqs = np.fft.fftfreq(sr, d=1.0 / sr)[:int(sr / 2)]
    s_power = np.abs(spectrum)

    peaks = signal.argrelmax(s_power, order=80)[0]
    peaks = peaks[(peaks >= fo)]

    return s_power, freqs, peaks


@st.cache
def move_ave(ts, win):
    ts_pad = np.pad(ts, [int(win / 2), int(win / 2)], 'reflect')
    return np.convolve(ts_pad, np.full(win, 1 / win), mode='same')[int(win / 2):-int(win / 2)]


def _set_block_container_style(
    max_width: int = GRAPH_WIDTH + 100,
    max_width_100_percent: bool = False,
    padding_top: int = 5,
    padding_right: int = 1,
    padding_left: int = 1,
    padding_bottom: int = 10,
):
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}

</style>
""",
        unsafe_allow_html=True,
    )

def main():
    st.title('Voice Analysis')
    link = '[<span class="hljs-string">Chemesim</span>](<span class="hljs-link">https://twitter.com/deiko_cs</span>)'
    st.markdown(link, unsafe<span class="hljs-emphasis">_allow_</span>html=True)
    st.subheader('使い方')
    st.write('1. 左のサイドバーを開いて音声を読み込む')
    st.write('2. サイドバーの設定から分析範囲を指定する')
    st.write('3. グラフや表に分析結果が表示される')
    st.write('※　１秒以上のモノラルwavファイルのみ使用可能')
    st.write('-----------------------------------')
    uploaded_file = st.sidebar.file_uploader("音声ファイル（モノラル、wavファイル）")

    if uploaded_file is not None:
        wav, sr = librosa.load(uploaded_file, sr=None)
        wav_seconds = int(len(wav) / sr)

        st.write(uploaded_file.name)
        st.audio(uploaded_file)

        st.sidebar.title('設定')
        tgt_ranges = st.sidebar.slider(
            "分析範囲（秒）", 0, wav_seconds, (0, wav_seconds))

        col1, col2 = st.columns(2)
        fig = go.Figure()
        x_wav = np.arange(len(wav)) / sr
        fig.add_trace(go.Scatter(y=wav[::HOP], name="wav"))
        fig.add_vrect(x0=int(tgt_ranges[0] * sr / HOP), x1=int(tgt_ranges[1] * sr / HOP),
                      fillcolor="LightSalmon", opacity=0.5, layer="below", line_width=0)
        fig.update_layout(title="【音声波形】", width=GRAPH_WIDTH, height=GRAPH_HEIGHT,
                          xaxis=dict(tickmode='array', tickvals=[1, int(len(wav[::HOP]) / 2), len(wav[::HOP])], ticktext=[str(0), str(int(wav_seconds / 2)), str(wav_seconds)], title="時間（秒）"))
        col1.plotly_chart(fig)

        wav_element = wav[tgt_ranges[0] * sr:tgt_ranges[1] * sr]

        # fo
        fo = calc_fo(wav_element)
        d_fo = fo[~np.isnan(fo)]
        ave_fo = np.average(d_fo)

        # spectrum
        s_power, freqs, peaks = calc_spectrum(wav_element, sr, ave_fo)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs, y=s_power, mode='lines', name=''),)
        fig.add_trace(go.Scatter(
            x=freqs[peaks[0:7]], y=s_power[peaks[0:7]], mode='markers', name='ピーク'))
        fig.update_layout(title="【周波数スペクトル】", width=GRAPH_WIDTH, height=GRAPH_HEIGHT,
                          xaxis=dict(title="周波数(Hz)",
                                     range=[0, 2000]),
                          yaxis=dict(title="強度"))
        col2.write(fig)
        odd = sum(s_power[peaks[1::2]])
        even = sum(s_power[peaks[2::2]])
        odd_per = odd * 100 / (odd + even)
        even_per = even * 100 / (odd + even)

        # hnr
        hnr = measurePitch(wav_element)

        st.title("分析結果")

        if hnr > 12:
            if ave_fo > 165:
                if odd_per > even_per+10:
                    st.write('この声は「エネルギー」、「元気」を感じます！')
                else:
                    st.write('この声は「ピュア」、「透明感」を感じます！')
            else:
                if odd_per > even_per+10:
                    st.write('この声は「リーダー」、「勇敢」を感じます！')
                else:
                    st.write('この声は「クール」、「信頼」を感じます！')
        else:
            if ave_fo > 165:
                if odd_per > even_per+10:
                    st.write('この声は「フレンドリー」、「愛嬌」を感じます！')
                else:
                    st.write('この声は「ソフト」、「甘い」を感じます！')
            else:
                if odd_per > even_per+10:
                    st.write('この声は「エレガント」、「妖艶」を感じます！')
                else:
                    st.write('この声は「ジェントル」、「貫禄」を感じます！')

        df = pd.DataFrame({"ファイル名": [uploaded_file.name],
                           "基本周波数（Hz）": [ave_fo],
                           "HNR（dB）": [hnr],
                           "奇数倍音（％）": [odd_per],
                           "偶数倍音（％）": [even_per]}
                          )
        st.dataframe(df)

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="result.csv">download</a>'
        st.markdown(f"csvファイルでダウンロード {href}", unsafe_allow_html=True)
        st.write('基本周波数とHNRは平均で計算しています。')

if __name__ == "__main__":
    _set_block_container_style()
    main()
