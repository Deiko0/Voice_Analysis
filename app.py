import streamlit as st
import streamlit.components.v1 as components
import librosa
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import base64
import parselmouth
from parselmouth.praat import call
from PIL import Image
import tempfile
import soundfile as sf
from pysndfx import AudioEffectsChain

HOP = 1000
GRAPH_WIDTH = 1200
GRAPH_HEIGHT = 300

st.set_page_config(
    page_title="Voice Analysis",
    menu_items={
        "Get Help": "https://twitter.com/deiko_cs",
        "Report a bug": "https://twitter.com/deiko_cs",
        "About": """
         # 声を分析してEQを提案するWebアプリ
         このWebアプリはアップロードした音声を分析することができます。周波数スペクトルや声の特徴、声のタイプを表示します。また、EQを提案して加工した音声をダウンロードできます。
         """,
    },
)

@st.cache_data
def measurePitch(wav):
    sound = parselmouth.Sound(wav)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = int(call(harmonicity, "Get mean", 0, 0))
    return hnr


@st.cache_data
def calc_spec(wav, sr):
    fo, voiced_flag, voiced_prob = librosa.pyin(
        wav, fmin=75, fmax=500)

    ave_fo = np.average(fo[voiced_flag])
    
    spectrum = np.abs(np.fft.fft(wav, sr)[: int(sr / 2)])
    freqs = np.fft.fftfreq(sr, d=1.0 / sr)[: int(sr / 2)]
    s_power = np.abs(spectrum) ** 2

    peaks = signal.argrelmax(s_power, order=70)[0]
    

    even = sum(s_power[peaks[1::2]])
    odd = sum(s_power[peaks[2::2]])
    if odd + even == 0:
        odd_per = 0
        even_per = 0
        return ave_fo, s_power, freqs, peaks, odd, even, odd_per, even_per
    else:
        odd_per = odd * 100 / (odd + even)
        even_per = even * 100 / (odd + even)

    return ave_fo, s_power, freqs, peaks, odd, even, odd_per, even_per


@st.cache_data
def draw_wave(wav, tgt_ranges, sr, wav_seconds):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=wav[::HOP], mode="lines", line=dict(color="#2584c1")))
    fig.add_vrect(
        x0=int(tgt_ranges[0] * sr / HOP),
        x1=int(tgt_ranges[1] * sr / HOP),
        fillcolor="#d89648",
        opacity=0.5,
        layer="below",
        line_width=0,
    )
    fig.update_layout(
        title="Waveform",
        height=GRAPH_HEIGHT,
        xaxis=dict(
            tickmode="array",
            tickvals=[1, int(len(wav[::HOP]) / 2), len(wav[::HOP])],
            ticktext=[str(0), str(int(wav_seconds / 2)), str(wav_seconds)],
            title="Time(s)",
            gridcolor="#e5edef",
            color="#20323e",
        ),
        yaxis=dict(gridcolor="#e5edef", color="#20323e", showticklabels=False),
        margin=dict(t=50, b=50, l=10, r=10),
        plot_bgcolor="#b7c3d1",
        paper_bgcolor="#e5edef",
        font_color="#20323e",
        font_size=20,
    )
    img = fig.to_image(format="png", width=600, height=525)
    return img


@st.cache_data
def draw_spectrum(freqs, s_power, peaks):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=freqs, y=s_power, mode="lines",
                   line=dict(color="#2584c1"))
    )
    fig.add_trace(
        go.Scatter(
            x=freqs[peaks[0:7]],
            y=s_power[peaks[0:7]],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=15),
            text=freqs[peaks[0:7]],
            marker=dict(color="#e3619f", size=10),
        )
    )
    fig.update_layout(
        title="Frequency Spectrum",
        height=GRAPH_HEIGHT,
        xaxis=dict(
            title="Frequency(Hz)", range=[0, 2000], gridcolor="#e5edef", color="#20323e"
        ),
        yaxis=dict(gridcolor="#e5edef", color="#20323e", showticklabels=False),
        showlegend=False,
        margin=dict(t=50, b=50, l=10, r=10),
        plot_bgcolor="#b7c3d1",
        paper_bgcolor="#e5edef",
        font_color="#20323e",
        font_size=20,
    )
    img = fig.to_image(format="png", width=600, height=525)
    return img


@st.cache_data
def draw_result(ave_fo, hnr, even_per, odd_per):
    fig = make_subplots(rows=3, cols=1)

    clip_ave_fo = np.clip(ave_fo, 70, 230)
    New_fo_Value = (((clip_ave_fo - 70) * 10) / 160) - 5
    if New_fo_Value > 0:
        fo_color = "#e3619f"
    else:
        fo_color = "#2584c1"

    fig.append_trace(
        go.Scatter(
            y=[""],
            x=[New_fo_Value],
            marker=dict(color=fo_color, size=40, symbol="diamond"),
        ),
        row=1,
        col=1,
    )
    fig.add_annotation(
        text="Low",
        xref="paper",
        yref="paper",
        x=0,
        y=0.86,
        showarrow=False,
        bgcolor="#e5edef",
        opacity=0.8,
        font_color="#20323e",
        font_size=30,
    )
    fig.add_annotation(
        text="High",
        xref="paper",
        yref="paper",
        x=1,
        y=0.86,
        showarrow=False,
        bgcolor="#e5edef",
        opacity=0.8,
        font_color="#20323e",
        font_size=30,
    )

    clip_hnr = np.clip(hnr, 9, 17)
    New_hnr_Value = (((clip_hnr - 9) * 10) / 8) - 5
    if New_hnr_Value > 0:
        hnr_color = "#e3619f"
    else:
        hnr_color = "#2584c1"

    fig.append_trace(
        go.Scatter(
            y=[""],
            x=[New_hnr_Value],
            marker=dict(color=hnr_color, size=40, symbol="diamond"),
        ),
        row=2,
        col=1,
    )
    fig.add_annotation(
        text="Husky",
        xref="paper",
        yref="paper",
        x=0,
        y=0.43,
        showarrow=False,
        bgcolor="#e5edef",
        opacity=0.8,
        font_size=30,
    )
    fig.add_annotation(
        text="Clear",
        xref="paper",
        yref="paper",
        x=1,
        y=0.43,
        showarrow=False,
        bgcolor="#e5edef",
        opacity=0.8,
        font_size=30,
    )

    fig.append_trace(
        go.Funnel(y=[""], x=[even_per], textinfo="text",
                  marker=dict(color="#2584c1")),
        row=3,
        col=1,
    )
    fig.append_trace(
        go.Funnel(y=[""], x=[odd_per], textinfo="text",
                  marker=dict(color="#e3619f")),
        row=3,
        col=1,
    )
    fig.add_annotation(
        text="Warm",
        xref="paper",
        yref="paper",
        x=0,
        y=0,
        showarrow=False,
        bgcolor="#e5edef",
        opacity=0.8,
        font_color="#20323e",
        font_size=30,
    )
    fig.add_annotation(
        text="Clarity",
        xref="paper",
        yref="paper",
        x=1,
        y=0,
        showarrow=False,
        bgcolor="#e5edef",
        opacity=0.8,
        font_color="#20323e",
        font_size=30,
    )
    fig.update_yaxes(gridcolor="#e5edef")
    fig.update_xaxes(dtick=1.25, showticklabels=False, gridcolor="#e5edef")
    fig.update_layout(
        xaxis=dict(range=[-5, 5]),
        xaxis2=dict(range=[-5, 5]),
        showlegend=False,
        margin=dict(t=0, b=0, l=10, r=10),
        plot_bgcolor="#b7c3d1",
        paper_bgcolor="#e5edef",
    )

    img = fig.to_image(format="png", width=600, height=350)

    return img


@st.cache_data
def calc_type(img_path):
    image = Image.open(img_path)
    twitter = (
        """
        <a href="http://twitter.com/intent/tweet" class="twitter-share-button"
        data-text="#レコメンドEQ #VoiceAnalysis"
        data-url="https://deiko0-voice-analysis-app-m0fgp5.streamlit.app"
        Tweet
        </a>
        <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
        """
    )
    return twitter, image


@st.cache_data
def get_binary_file_downloader_html(bin_file, file_label='File', extension=""):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}_eq.wav">Download</a>'
    return href


@st.cache_data
def eq_recommended(filename, wav, sr, ave_fo, peaks, eq_gain):
    eq2_peaks = peaks[(peaks >= 1500) & (peaks <= 3000)]
    eq1_peaks = peaks[(peaks >= 400) & (peaks <= 500)]

    if len(eq2_peaks) == 0:
        eq2 = 2500
    else:
        eq2 = int(np.median(eq2_peaks))

    if len(eq1_peaks) == 0:
        eq1 = 450
    else:
        eq1 = int(np.median(eq1_peaks))

    low = int(ave_fo-5)

    fx = AudioEffectsChain().highpass(low, q=1/np.sqrt(2)).equalizer(eq1, q=4.0,
                                                                     db=eq_gain*-1).equalizer(eq2, q=0.46, db=eq_gain)
    eq_wav = fx(wav)

    fp = tempfile.NamedTemporaryFile()
    sf.write(fp.name, eq_wav, sr, format="wav", subtype="PCM_24")
    href2 = get_binary_file_downloader_html(fp.name, filename, ".wav")

    return eq_wav, low, eq1, eq2, href2


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
    st.title("Voice Analysis")
    st.write("create by Deiko")
    st.markdown("---")
    st.subheader("How to use")
    col1, col2 = st.columns(2)
    uploaded_file = col2.file_uploader("＊1秒以上の.wavのみ対応")

    col1.write("①「Browse files」から音声ファイルを読み込む")
    col1.write("② ピンクのスライドバーで分析範囲を指定する")
    col1.write("③ グラフや表に分析結果が表示される")
    st.markdown("---")

    if uploaded_file is not None:
        if not uploaded_file.name.endswith(".wav"):
            st.error("このファイルのフォーマットに対応していません！.wavファイルを読み込んでください！", icon="😵")
        else:
            wav, sr = librosa.load(uploaded_file, sr=None)
            wav_seconds = int(len(wav) / sr)

            col2.audio(wav, sample_rate=sr)

            tgt_ranges = col2.slider(
                "分析範囲（秒）", 0, wav_seconds, (0, wav_seconds))

            wav_element = wav[tgt_ranges[0] * sr: tgt_ranges[1] * sr]

            # spec
            ave_fo, s_power, freqs, peaks, odd, even, odd_per, even_per = calc_spec(
                wav_element, sr
            )

            col3, col4 = st.columns(2)

            wave_img = draw_wave(wav, tgt_ranges, sr, wav_seconds)
            col3.image(wave_img)

            spectrum_img = draw_spectrum(freqs, s_power, peaks)
            col4.image(spectrum_img)

            if tgt_ranges == (0, 0):
                st.error("分析範囲が0秒です！設定から分析範囲を指定し直してください！", icon="😵")
            elif odd_per + even_per == 0:
                st.error("倍音が検出できません！設定から分析範囲を指定し直すか、別のファイルを読み込んでください！", icon="😵")
            else:
                # hnr
                hnr = measurePitch(wav_element)

                st.header("Result")
                col5, col6 = st.columns(2)

                result_img = draw_result(ave_fo, hnr, even_per, odd_per)
                col5.image(result_img)

                if hnr > 13:
                    if ave_fo > 150:
                        if odd_per > even_per + 10:
                            type = "あなたの声は【元気】、【エネルギー】タイプです！"
                            img_path = "images/energy.png"
                        else:
                            type = "あなたの声は【透明】、【ピュア】タイプです！"
                            img_path = "images/pure.png"
                    else:
                        if odd_per > even_per + 10:
                            type = "あなたの声は【勇敢】、【リーダー】タイプです！"
                            img_path = "images/leader.png"
                        else:
                            type = "あなたの声は【信頼】、【クール】タイプです！"
                            img_path = "images/cool.png"
                else:
                    if ave_fo > 150:
                        if odd_per > even_per + 10:
                            type = "あなたの声は【愛嬌】、【フレンド】タイプです！"
                            img_path = "images/friend.png"
                        else:
                            type = "あなたの声は【甘い】、【ソフト】タイプです！"
                            img_path = "images/soft.png"
                    else:
                        if odd_per > even_per + 10:
                            type = "あなたの声は【妖艶】、【エレガント】タイプです！"
                            img_path = "images/elegant.png"
                        else:
                            type = "あなたの声は【貫禄】、【ジェントル】タイプです！"
                            img_path = "images/gentle.png"

                twitter, image = calc_type(img_path)
                col6.image(image)
                df = pd.DataFrame(
                    {
                        "ファイル名": [uploaded_file.name],
                        "基本周波数（Hz）": [ave_fo],
                        "HNR（dB）": [hnr],
                        "奇数倍音（％）": [odd_per],
                        "偶数倍音（％）": [even_per]
                    }
                )
                st.dataframe(df)

                filename = uploaded_file.name.removesuffix('.wav')
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.csv">Download</a>'
                st.markdown(
                    f'<span style="font-size:16px">csvファイルでダウンロード▶︎ {href}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<span style="font-size:16px">基本周波数とHNRは平均で計算しています。</span>',
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                st.subheader("Recommended EQ")
                st.write("分析結果を元におすすめの音声EQを提案します！（ナレーション向け）")
                eq_gain = st.slider("レコメンドEQのGain適用度", 1, 5, 1)
                eq_wav, low, eq1, eq2, href2 = eq_recommended(
                    filename, wav, sr, ave_fo, peaks, eq_gain)
                eq_df = pd.DataFrame(
                    data=np.array([['', 'Peaking', 'Peaking'], [low, eq1, eq2], [
                                  '12dB/oct', 4.0, 0.46], ['', eq_gain*-1, eq_gain]]),
                    index=['タイプ', '周波数（Hz）', 'Q', 'Gain（dB）'],
                    columns=['High Pass Filter',
                             'Low Mid Frequency', 'High Mid Frequency']
                )
                st.dataframe(eq_df)

                col7, col8 = st.columns(2)

                col7.write("[Before]")
                col7.audio(wav, sample_rate=sr)
                col8.write("[After]")
                col8.audio(eq_wav, sample_rate=sr)

                col8.markdown(
                    f'<span style="font-size:16px">wavファイルでダウンロード▶︎ {href2}</span>', unsafe_allow_html=True)
                components.html(twitter)


if __name__ == "__main__":
    _set_block_container_style()
    main()
