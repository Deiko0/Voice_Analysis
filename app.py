import streamlit as st
import streamlit.components.v1 as components
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
GRAPH_HEIGHT = 300

st.set_page_config(
    page_title="Voice Analysis",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://twitter.com/deiko_cs',
        'Report a bug': "https://twitter.com/deiko_cs",
        'About': """
         # 声の分析ツール
         このツールはアップロードした音声を分析して、グラフや声のタイプを表示します。
         """
    })

config = dict(
    {'modeBarButtonsToRemove': [
        'toImage',
        'zoom2d',
        'pan2d',
        'select2d',
        'lasso2d',
        'zoomIn2d',
        'zoomOut2d',
        'autoScale2d',
        'resetScale2d',
    ],
        'displaylogo': False}
)


@st.cache
def measurePitch(wav):
    sound = parselmouth.Sound(wav)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    return hnr


@st.cache
def calc_spec(wav, sr):
    fo, voiced_flag, voiced_prob = librosa.pyin(wav, 80, 2000)
    d_fo = fo[~np.isnan(fo)]
    ave_fo = np.average(d_fo)

    spectrum = np.abs(np.fft.fft(wav, sr)[:int(sr / 2)])
    freqs = np.fft.fftfreq(sr, d=1.0 / sr)[:int(sr / 2)]
    s_power = np.abs(spectrum)

    peaks = signal.argrelmax(s_power, order=80)[0]
    peaks = peaks[(peaks >= ave_fo)]

    odd = sum(s_power[peaks[1::2]])
    even = sum(s_power[peaks[2::2]])
    if odd + even == 0:
        odd_per = 0
        even_per = 0
        return odd, even, odd_per, even_per
    else:
        odd_per = odd * 100 / (odd + even)
        even_per = even * 100 / (odd + even)

    return ave_fo, s_power, freqs, peaks, odd, even, odd_per, even_per


@st.cache
def draw_hnr(fig, hnr):
    clip_hnr = np.clip(hnr, 7, 17)
    New_hnr_Value = (((clip_hnr - 7) * 10) / 10) - 5
    if New_hnr_Value > 0:
        hnr_color = '#e3619f'
    else:
        hnr_color = '#2584c1'

    fig.add_trace(go.Scatter(y=[''], x=[New_hnr_Value], marker=dict(
        color=hnr_color, size=20, symbol='diamond')),)
    fig.add_annotation(text='ハスキー', xref="paper", yref="paper",
                       x=0, y=0.5, showarrow=False, bgcolor="#e5edef",
                       opacity=0.8, font=dict(
                           family="monospace",
                           color="#20323e",
                           size=16
                       ))
    fig.add_annotation(text='クリア', xref="paper", yref="paper",
                       x=1, y=0.5, showarrow=False, bgcolor="#e5edef",
                       opacity=0.8, font=dict(
                           family="monospace",
                           color="#20323e",
                           size=16
                       ))
    fig.update_yaxes(gridcolor='#e5edef')
    fig.update_xaxes(dtick=1.25, showticklabels=False,
                     gridcolor='#e5edef')
    fig.update_layout(height=GRAPH_HEIGHT / 6, xaxis=dict(
        range=[-5, 5]), showlegend=False, hovermode=False, margin=dict(t=0, b=0, l=10, r=10), plot_bgcolor="#b7c3d1")
    return fig


@st.cache
def draw_fo(fig, ave_fo):
    clip_ave_fo = np.clip(ave_fo, 80, 250)
    New_fo_Value = (((clip_ave_fo - 80) * 10) / 170) - 5
    if New_fo_Value > 0:
        fo_color = '#e3619f'
    else:
        fo_color = '#2584c1'

    fig.add_trace(go.Scatter(y=[''], x=[New_fo_Value], marker=dict(
        color=fo_color, size=20, symbol='diamond')),)
    fig.add_annotation(text='低音', xref="paper", yref="paper",
                       x=0, y=0.5, showarrow=False, bgcolor="#e5edef",
                       opacity=0.8, font=dict(
                           family="monospace",
                           color="#20323e",
                           size=16
                       ))
    fig.add_annotation(text='高音', xref="paper", yref="paper",
                       x=1, y=0.5, showarrow=False, bgcolor="#e5edef",
                       opacity=0.8, font=dict(
                           family="monospace",
                           color="#20323e",
                           size=16
                       ))
    fig.update_yaxes(gridcolor='#e5edef')
    fig.update_xaxes(dtick=1.25, showticklabels=False,
                     gridcolor='#e5edef')
    fig.update_layout(height=GRAPH_HEIGHT / 6, xaxis=dict(
        range=[-5, 5]), showlegend=False, hovermode=False, margin=dict(t=0, b=0, l=10, r=10), plot_bgcolor="#b7c3d1")
    return fig


@st.cache
def draw_per(fig, even_per, odd_per):
    fig.add_trace(go.Funnel(y=[''], x=[even_per], textinfo='text', marker=dict(
        color='#2584c1')))
    fig.add_trace(go.Funnel(y=[''], x=[odd_per], textinfo='text', marker=dict(
        color='#e3619f')))
    fig.add_annotation(text='柔和', xref="paper", yref="paper",
                       x=0, y=0.5, showarrow=False, bgcolor="#e5edef",
                       opacity=0.8, font=dict(
                           family="monospace",
                           color="#20323e",
                           size=16
                       ))
    fig.add_annotation(text='明瞭', xref="paper", yref="paper",
                       x=1, y=0.5, showarrow=False, bgcolor="#e5edef",
                       opacity=0.8, font=dict(
                           family="monospace",
                           color="#20323e",
                           size=16
                       ))
    fig.update_layout(height=GRAPH_HEIGHT / 6, showlegend=False,
                      hovermode=False, margin=dict(t=0, b=0, l=10, r=10), plot_bgcolor="#b7c3d1")
    return fig


@st.cache
def calc_type(type):
    twitter_type = """
        <meta name=”twitter:card” content=”summary_large_image” />
        <meta name=”twitter:site” content=”@deiko_cs” />
        <meta name=”twitter:domain” content=”deiko0-voice-analysis-app-m0fgp5.streamlit.app” />
        <meta name=”twitter:title” content=”Voice Analysis” />
        <meta name=”twitter:description” content=”声を分析するWebツール[…]” />
        <meta name="twitter:image" content="" />
        <a href="https://twitter.com/intent/tweet" class="twitter-share-button"
        data-text=分析の結果、""" + type + """#あなたの声は何タイプ、#VoiceAnalysis
        data-url="https://deiko0-voice-analysis-app-m0fgp5.streamlit.app"
        Tweet
        </a>
        <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
        """
    return twitter_type


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
    st.write('create by Deiko')
    st.markdown("---")
    st.sidebar.title('Voice Analysis')
    uploaded_file = st.sidebar.file_uploader('＊1秒以上のwav、モノラル音源')

    st.sidebar.title('使い方')
    st.sidebar.write('1.「Browse files」から音声ファイルを読み込む')
    st.sidebar.write('2.「設定」から分析範囲を指定する')
    st.sidebar.write('3.グラフや表に分析結果が表示される')
    st.sidebar.markdown("---")

    if uploaded_file is not None:
        wav, sr = librosa.load(uploaded_file, sr=None)
        wav = librosa.to_mono(wav)
        wav_seconds = int(len(wav) / sr)

        st.write('ファイル名：' + uploaded_file.name)
        st.audio(uploaded_file)

        st.sidebar.title('設定')
        tgt_ranges = st.sidebar.slider(
            "分析範囲（秒）", 0, wav_seconds, (0, wav_seconds))
        st.sidebar.markdown("---")

        wav_element = wav[tgt_ranges[0] * sr:tgt_ranges[1] * sr]

        # spec
        ave_fo, s_power, freqs, peaks, odd, even, odd_per, even_per = calc_spec(
            wav_element, sr)

        col1, col2 = st.columns(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=wav[::HOP], mode='lines', line=dict(color="#2584c1")))
        fig.add_vrect(x0=int(tgt_ranges[0] * sr / HOP), x1=int(tgt_ranges[1] * sr / HOP),
                      fillcolor="#d89648", opacity=0.5, layer="below", line_width=0)
        fig.update_yaxes(gridcolor='#e5edef', color="#20323e")
        fig.update_xaxes(gridcolor='#e5edef', color="#20323e")
        fig.update_layout(title="【音声波形】", height=GRAPH_HEIGHT,
                          xaxis=dict(tickmode='array', tickvals=[1, int(len(wav[::HOP]) / 2), len(wav[::HOP])], ticktext=[
                                     str(0), str(int(wav_seconds / 2)), str(wav_seconds)], title="時間（秒）"),
                          margin=dict(t=50, b=0, l=10, r=10),
                          plot_bgcolor="#b7c3d1"
                          )
        col1.plotly_chart(fig, use_container_width=True, **{'config': config})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs, y=s_power,
                      mode='lines', line=dict(color="#2584c1")))
        fig.add_trace(go.Scatter(
            x=freqs[peaks[0:7]], y=s_power[peaks[0:7]], mode='markers', marker=dict(
                color='#e3619f')))
        fig.update_yaxes(gridcolor='#e5edef', color="#20323e")
        fig.update_xaxes(gridcolor='#e5edef', color="#20323e")
        fig.update_layout(title="【周波数スペクトル】", height=GRAPH_HEIGHT,
                          xaxis=dict(title="周波数（Hz）",
                                     range=[0, 2000]),
                          showlegend=False,
                          margin=dict(t=50, b=0, l=10, r=10),
                          plot_bgcolor="#b7c3d1"
                          )
        col2.plotly_chart(fig, use_container_width=True, **{'config': config})

        if tgt_ranges == (0, 0):
            st.error('分析範囲が0秒です！設定から分析範囲を指定し直してください！', icon='😵')
        elif odd_per + even_per == 0:
            st.error('倍音が検出できません！設定から分析範囲を指定し直すか、別のファイルを読み込んでください！', icon='😵')
        else:
            # hnr
            hnr = measurePitch(wav_element)

            st.title("分析結果")
            col3, col4 = st.columns(2)

            fig = go.Figure()
            fig = draw_fo(fig, ave_fo)
            col3.plotly_chart(fig, use_container_width=True,
                              **{'config': config})

            fig = go.Figure()
            fig = draw_hnr(fig, hnr)
            col3.plotly_chart(fig, use_container_width=True,
                              **{'config': config})

            fig = go.Figure()
            fig = draw_per(fig, even_per, odd_per)
            col3.plotly_chart(fig, use_container_width=True,
                              **{'config': config})

            if hnr > 12:
                if ave_fo > 165:
                    if odd_per > even_per + 10:
                        type = '高音とクリアと明瞭を読み取りました！あなたの声は【元気】、【エネルギー】タイプです！'
                        twitter_type = calc_type(type)
                        col4.write(type)
                        components.html(twitter_type)
                    else:
                        type = '高音とクリアと柔和を読み取りました！あなたの声は【透明】、【ピュア】タイプです！'
                        twitter_type = calc_type(type)
                        col4.write(type)
                        components.html(twitter_type)
                else:
                    if odd_per > even_per + 10:
                        type = '低音とクリアと明瞭を読み取りました！あなたの声は【勇敢】、【リーダー】タイプです！'
                        twitter_type = calc_type(type)
                        col4.write(type)
                        components.html(twitter_type)
                    else:
                        type = '低音とクリアと柔和を読み取りました！あなたの声は【信頼】、【クール】タイプです！'
                        twitter_type = calc_type(type)
                        col4.write(type)
                        components.html(twitter_type)
            else:
                if ave_fo > 165:
                    if odd_per > even_per + 10:
                        type = '高音とハスキーと明瞭を読み取りました！あなたの声は【愛嬌】、【フレンドリー】タイプです！'
                        twitter_type = calc_type(type)
                        col4.write(type)
                        components.html(twitter_type)
                    else:
                        type = '高音とハスキーと柔和を読み取りました！あなたの声は【甘い】、【ソフト】タイプです！'
                        twitter_type = calc_type(type)
                        col4.write(type)
                        components.html(twitter_type)
                else:
                    if odd_per > even_per + 10:
                        type = '低音とハスキーと明瞭を読み取りました！あなたの声は【妖艶】、【エレガント】タイプです！'
                        twitter_type = calc_type(type)
                        col4.write(type)
                        components.html(twitter_type)
                    else:
                        type = '低音とハスキーと柔和を読み取りました！あなたの声は【貫禄】、【ジェントル】タイプです！'
                        twitter_type = calc_type(type)
                        col4.write(type)
                        components.html(twitter_type)

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
            st.markdown(
                f'<span style="font-family:monospace;font-size:16px">csvファイルでダウンロード {href}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<span style="font-family:monospace;font-size:16px">基本周波数とHNRは平均で計算しています。</span>', unsafe_allow_html=True)


if __name__ == "__main__":
    _set_block_container_style()
    main()
