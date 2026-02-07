"""
Streamlit UI for Productivity Monitoring System
Provides web interface for attention tracking with analytics.

Author: Senior Developer
Date: 2026-02-08
"""

import streamlit as st
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
from main_streamlit import ProductivityMonitorStreamlit

# Page configuration
st.set_page_config(
    page_title="Productivity Monitor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'monitor' not in st.session_state:
    st.session_state.monitor = None
if 'is_tracking' not in st.session_state:
    st.session_state.is_tracking = False
if 'session_ended' not in st.session_state:
    st.session_state.session_ended = False
if 'session_summary' not in st.session_state:
    st.session_state.session_summary = None


def initialize_monitor():
    """Initialize the productivity monitor."""
    try:
        if st.session_state.monitor is None:
            st.session_state.monitor = ProductivityMonitorStreamlit()
        return True
    except Exception as e:
        st.error(f"Error initializing monitor: {e}")
        return False


def start_tracking():
    """Start tracking session."""
    if initialize_monitor():
        try:
            st.session_state.monitor.start_tracking()
            st.session_state.is_tracking = True
            st.session_state.session_ended = False
            st.session_state.session_summary = None
            st.success("✅ Tracking started! Please calibrate for best results.")
        except Exception as e:
            st.error(f"Error starting tracking: {e}")


def calibrate_system():
    """Calibrate the system."""
    if st.session_state.monitor is not None and st.session_state.is_tracking:
        st.session_state.monitor.calibrate()
        st.success("✅ Calibration triggered! Look at the camera in your normal working position.")
    else:
        st.warning("⚠️ Please start tracking first!")


def stop_tracking():
    """Stop tracking session."""
    if st.session_state.monitor is not None and st.session_state.is_tracking:
        st.session_state.monitor.stop_tracking()
        st.session_state.session_summary = st.session_state.monitor.get_session_summary()
        st.session_state.monitor.save_session()
        st.session_state.is_tracking = False
        st.session_state.session_ended = True
        st.success("✅ Session ended and saved!")
    else:
        st.warning("⚠️ No active tracking session!")


def load_session_history():
    """Load session history from CSV."""
    csv_file = "session_history.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            return df
        except Exception as e:
            st.error(f"Error loading session history: {e}")
            return None
    return None


def create_timeline_chart(timeline_data):
    """Create time-series chart of attention over time."""
    if not timeline_data or len(timeline_data) == 0:
        return None
    
    # Prepare data for plotting
    timestamps = [point['timestamp'] for point in timeline_data]
    states = [point['state'] for point in timeline_data]
    is_attentive = [1 if point['is_attentive'] else 0 for point in timeline_data]
    
    # Convert timestamps to relative seconds from start
    start_time = timestamps[0]
    relative_times = [(t - start_time).total_seconds() / 60 for t in timestamps]  # Convert to minutes
    
    df = pd.DataFrame({
        'Time (minutes)': relative_times,
        'Attentive': is_attentive,
        'State': states
    })
    
    # Create figure
    fig = go.Figure()
    
    # Add area chart
    fig.add_trace(go.Scatter(
        x=df['Time (minutes)'],
        y=df['Attentive'],
        mode='lines',
        name='Attention',
        fill='tozeroy',
        line=dict(color='#2ecc71', width=2),
        fillcolor='rgba(46, 204, 113, 0.3)'
    ))
    
    fig.update_layout(
        title="Attention Timeline",
        xaxis_title="Time (minutes)",
        yaxis_title="Status",
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['Distracted', 'Attentive']
        ),
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    return fig


def create_state_breakdown_pie(state_breakdown):
    """Create pie chart for state breakdown."""
    if not state_breakdown:
        return None
    
    # Prepare data
    states = list(state_breakdown.keys())
    durations = list(state_breakdown.values())
    
    # Color mapping
    color_map = {
        'ATTENTIVE': '#2ecc71',
        'AWAY': '#e74c3c',
        'HEAD DISTRACTED': '#f39c12',
        'EYES DISTRACTED': '#3498db'
    }
    
    colors = [color_map.get(state, '#95a5a6') for state in states]
    
    fig = go.Figure(data=[go.Pie(
        labels=states,
        values=durations,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Time Distribution by State",
        height=400,
        showlegend=True
    )
    
    return fig


def display_session_analytics(summary):
    """Display comprehensive session analytics."""
    if summary is None:
        st.info("Complete a tracking session to see analytics here.")
        return
    
    # Header
    st.markdown("### 📊 Session Summary")
    
    # Session info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Session Start", summary['session_start'].strftime('%I:%M:%S %p'))
    with col2:
        st.metric("Session End", summary['session_end'].strftime('%I:%M:%S %p'))
    with col3:
        st.metric("Total Duration", summary['total_duration_formatted'])
    
    st.markdown("---")
    
    # Key metrics
    st.markdown("### 🎯 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #2ecc71; margin: 0;">Attentive Time</h4>
            <h2 style="margin: 0.5rem 0;">{summary['attentive_formatted']}</h2>
            <p style="margin: 0; color: #666;">{summary['attentive_seconds']:.1f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #e74c3c; margin: 0;">Distracted Time</h4>
            <h2 style="margin: 0.5rem 0;">{summary['distracted_formatted']}</h2>
            <p style="margin: 0; color: #666;">{summary['distracted_seconds']:.1f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        score = summary['attentiveness_score']
        score_color = "#2ecc71" if score >= 70 else "#f39c12" if score >= 50 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: {score_color}; margin: 0;">Attentiveness Score</h4>
            <h2 style="margin: 0.5rem 0;">{score:.1f}%</h2>
            <p style="margin: 0; color: #666;">Overall performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #3498db; margin: 0;">Longest Focus</h4>
            <h2 style="margin: 0.5rem 0;">{summary['longest_attentive_formatted']}</h2>
            <p style="margin: 0; color: #666;">Continuous period</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Timeline chart
        if summary.get('timeline'):
            fig_timeline = create_timeline_chart(summary['timeline'])
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("Timeline data not available for this session.")
    
    with col2:
        # Pie chart
        if summary.get('state_breakdown'):
            fig_pie = create_state_breakdown_pie(summary['state_breakdown'])
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed breakdown
    st.markdown("### 📋 Detailed Breakdown")
    
    if summary.get('state_breakdown'):
        breakdown_data = []
        total_seconds = summary['total_duration_seconds']
        
        for state, duration in summary['state_breakdown'].items():
            percentage = (duration / total_seconds * 100) if total_seconds > 0 else 0
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            breakdown_data.append({
                'State': state,
                'Duration': formatted,
                'Seconds': f"{duration:.2f}",
                'Percentage': f"{percentage:.2f}%"
            })
        
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
    
    # Additional insights
    st.markdown("### 💡 Focus Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Longest Attentive Period:** {summary['longest_attentive_formatted']}  
        **Longest Distracted Period:** {summary['longest_distracted_formatted']}
        """)
    
    with col2:
        if summary.get('state_history'):
            transitions = len([h for h in summary['state_history'] if h['state'] != 'AWAY'])
            st.markdown(f"""
            **Total State Changes:** {transitions}  
            **Session Quality:** {'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Fair' if score >= 40 else 'Needs Improvement'}
            """)


def display_historical_sessions():
    """Display historical session data."""
    st.markdown("### 📚 Session History")
    
    df_history = load_session_history()
    
    if df_history is None or len(df_history) == 0:
        st.info("No historical sessions found. Complete a session to build your history!")
        return
    
    # Session selector
    session_options = []
    for idx, row in df_history.iterrows():
        session_label = f"Session {idx + 1}: {row['session_start']} ({row['total_duration_formatted']})"
        session_options.append(session_label)
    
    selected_session = st.selectbox(
        "Select a session to view:",
        options=range(len(session_options)),
        format_func=lambda x: session_options[x]
    )
    
    if selected_session is not None:
        session_data = df_history.iloc[selected_session]
        
        # Display selected session
        st.markdown("---")
        st.markdown(f"#### Session Details: {session_data['session_start']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Duration", session_data['total_duration_formatted'])
            st.metric("Attentive Time", session_data['attentive_time_formatted'])
        
        with col2:
            st.metric("Distracted Time", session_data['distracted_time_formatted'])
            st.metric("Score", f"{session_data['attentiveness_score']:.1f}%")
        
        with col3:
            st.metric("Longest Focus", session_data['longest_attentive_period_formatted'])
            st.metric("Longest Distraction", session_data['longest_distracted_period_formatted'])
        
        # State breakdown for historical session
        st.markdown("##### State Breakdown")
        
        # Extract state data
        state_cols = [col for col in df_history.columns if col.endswith('_seconds') and not col.startswith('total') 
                      and not col.startswith('attentive') and not col.startswith('distracted') 
                      and not col.startswith('longest')]
        
        if state_cols:
            state_data = {}
            for col in state_cols:
                state_name = col.replace('_seconds', '').replace('_', ' ').upper()
                if pd.notna(session_data[col]) and session_data[col] > 0:
                    state_data[state_name] = session_data[col]
            
            if state_data:
                fig_pie_hist = create_state_breakdown_pie(state_data)
                if fig_pie_hist:
                    st.plotly_chart(fig_pie_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Show all sessions summary
    st.markdown("#### All Sessions Overview")
    
    # Prepare summary data
    summary_df = df_history[['session_start', 'total_duration_formatted', 
                             'attentiveness_score', 'attentive_time_formatted', 
                             'distracted_time_formatted']].copy()
    summary_df.columns = ['Date/Time', 'Duration', 'Score (%)', 'Attentive', 'Distracted']
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Trend analysis if multiple sessions
    if len(df_history) > 1:
        st.markdown("#### 📈 Score Trend")
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=list(range(1, len(df_history) + 1)),
            y=df_history['attentiveness_score'],
            mode='lines+markers',
            name='Attentiveness Score',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
        
        fig_trend.update_layout(
            title="Attentiveness Score Over Sessions",
            xaxis_title="Session Number",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 100]),
            height=350
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)


# Main UI
st.markdown('<p class="main-header">👁️ Productivity Monitor</p>', unsafe_allow_html=True)

# Control panel
st.markdown("### 🎮 Control Panel")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Start Tracking", type="primary", disabled=st.session_state.is_tracking, use_container_width=True):
        start_tracking()

with col2:
    if st.button("⚙️ Calibrate", disabled=not st.session_state.is_tracking, use_container_width=True):
        calibrate_system()

with col3:
    if st.button("🛑 Stop Tracking", disabled=not st.session_state.is_tracking, use_container_width=True):
        stop_tracking()

# Status indicator
if st.session_state.is_tracking:
    st.success("🟢 **Status:** Tracking Active")
    
    if st.session_state.monitor is not None:
        # Show current stats
        current_duration = st.session_state.monitor.analytics.get_total_duration()
        current_score = st.session_state.monitor.analytics.get_attentiveness_score()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Session Duration", 
                     st.session_state.monitor.analytics.format_duration(current_duration))
        with col2:
            st.metric("Current Score", f"{current_score:.1f}%")
else:
    st.info("⚪ **Status:** Not Tracking")

st.markdown("---")

# Single Analytics Tab
st.markdown("### 📊 Session Analytics")

# Show instructions if not tracking and no session ended yet
if not st.session_state.is_tracking and not st.session_state.session_ended:
    st.info("👆 Click **Start Tracking** to begin monitoring. A camera window will open separately.")
    st.markdown("""
    **How it works:**
    1. Click **Start Tracking** - a camera window will open
    2. Click **Calibrate** button here while looking at the camera
    3. The system tracks your attention in the camera window
    4. Click **Stop Tracking** or close the camera window to see analytics
    
    **Camera Window Controls:**
    - Press **Q** in the camera window to quit
    - Or click the **X** button to close the window
    - Or use the **Stop Tracking** button above
    """)
    st.markdown("---")

# Show current session analytics if just ended
if st.session_state.session_ended and st.session_state.session_summary is not None:
    st.markdown("#### 🎯 Current Session Results")
    display_session_analytics(st.session_state.session_summary)
    st.markdown("---")

# Show historical sessions
display_historical_sessions()

# Auto-refresh to check if OpenCV window was closed
if st.session_state.is_tracking and st.session_state.monitor is not None:
    # Check if tracking is still active (OpenCV window might be closed)
    if not st.session_state.monitor.is_tracking_active():
        # User closed the OpenCV window - stop tracking and show analytics
        st.session_state.session_summary = st.session_state.monitor.get_session_summary()
        st.session_state.monitor.save_session()
        st.session_state.is_tracking = False
        st.session_state.session_ended = True
        st.rerun()
    else:
        # Still tracking - refresh to update stats
        time.sleep(0.5)
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Productivity Monitor v1.0 | Track your focus, improve your productivity</p>
</div>
""", unsafe_allow_html=True)
