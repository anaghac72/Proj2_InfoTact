import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="📚 Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .book-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .similarity-score {
        background: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None

# Load your data from the saved pickle file
@st.cache_data
def load_data():
    try:
        # First, try to load from saved pickle file
        with open('dashboard_data/book_data.pkl', 'rb') as f:
            data = pickle.load(f)
            st.success("✅ Loaded data from saved file!")
            return data['pt'], data['similarities']
    except FileNotFoundError:
        # If no saved data, show instructions to user
        st.warning("⚠️ No saved data found. Please save your data first!")
        
        with st.expander("📝 Instructions to save your data", expanded=True):
            st.markdown("""
            **Add this code to your Jupyter notebook after creating your `pt` and `similarities`:**
            
            ```python
            import pickle
            import os
            
            # Create directory if it doesn't exist
            os.makedirs('dashboard_data', exist_ok=True)
            
            # Save your data
            with open('dashboard_data/book_data.pkl', 'wb') as f:
                pickle.dump({
                    'pt': pt, 
                    'similarities': similarities
                }, f)
            
            print("Data saved successfully!")
            ```
            
            **Then refresh this dashboard!**
            """)
        
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Recommendation function
def get_recommendations(book_name, pt, similarities, num_recommendations=10):
    if book_name in pt.index:
        index = np.where(pt.index == book_name)[0][0]
        similar_books_list = sorted(
            list(enumerate(similarities[index])), 
            key=lambda x: x[1], 
            reverse=True
        )[1:num_recommendations+1]
        
        recommendations = []
        for book_idx, similarity_score in similar_books_list:
            recommendations.append({
                'book': pt.index[book_idx],
                'similarity': round(similarity_score, 4)
            })
        
        return recommendations
    else:
        return None

# Load data
pt, similarities = load_data()

if pt is not None and similarities is not None:
    # Header
    st.markdown('<h1 class="main-header">📚 Book Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Search Options")
        
        # Book selection
        selected_book = st.selectbox(
            "Select a book:",
            options=[""] + list(pt.index),
            index=0,
            help="Choose a book to get recommendations"
        )
        
        # Number of recommendations
        num_recs = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=15,
            value=10,
            help="How many book recommendations do you want?"
        )
        
        # Search button
        search_clicked = st.button("🚀 Get Recommendations", type="primary")
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Dataset Info")
        st.info(f"Total books: {len(pt.index)}")
        
        # Sample books
        st.subheader("📖 Sample Books")
        for book in pt.index[:5]:
            st.write(f"• {book[:50]}{'...' if len(book) > 50 else ''}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if search_clicked and selected_book:
            with st.spinner("Finding similar books..."):
                recommendations = get_recommendations(selected_book, pt, similarities, num_recs)
                
                if recommendations:
                    st.session_state.recommendations = recommendations
                    st.session_state.selected_book = selected_book
                    
                    st.success(f"Found {len(recommendations)} recommendations for '{selected_book}'")
                    
                    # Display recommendations
                    st.subheader(f"📚 Recommendations for: {selected_book}")
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.container():
                            col_book, col_score = st.columns([4, 1])
                            
                            with col_book:
                                st.markdown(f"""
                                <div class="book-card">
                                    <strong>{i}. {rec['book']}</strong>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_score:
                                st.markdown(f"""
                                <div style="text-align: center; padding-top: 1rem;">
                                    <span class="similarity-score">{rec['similarity']:.3f}</span>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.error("Book not found in our database!")
        
        elif st.session_state.recommendations:
            # Show previous recommendations
            st.subheader(f"📚 Previous Recommendations for: {st.session_state.selected_book}")
            
            for i, rec in enumerate(st.session_state.recommendations, 1):
                with st.container():
                    col_book, col_score = st.columns([4, 1])
                    
                    with col_book:
                        st.markdown(f"""
                        <div class="book-card">
                            <strong>{i}. {rec['book']}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_score:
                        st.markdown(f"""
                        <div style="text-align: center; padding-top: 1rem;">
                            <span class="similarity-score">{rec['similarity']:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            # Welcome message
            st.markdown("""
            ## 👋 Welcome to the Book Recommendation System!
            
            This system helps you discover new books based on your preferences using machine learning algorithms.
            
            ### How it works:
            1. **Select a book** from the dropdown in the sidebar
            2. **Choose** how many recommendations you want
            3. **Click** "Get Recommendations" to see similar books
            4. **Explore** the results with similarity scores
            
            ### Features:
            - ✨ Intelligent book recommendations
            - 📊 Similarity scores for each recommendation
            - 🎯 Customizable number of results
            - 📱 Responsive design
            
            **Get started by selecting a book from the sidebar!**
            """)
    
    with col2:
        if st.session_state.recommendations:
            # Visualization of similarity scores
            st.subheader("📊 Similarity Scores")
            
            # Create a bar chart
            books = [rec['book'][:20] + '...' if len(rec['book']) > 20 else rec['book'] 
                    for rec in st.session_state.recommendations[:10]]
            scores = [rec['similarity'] for rec in st.session_state.recommendations[:10]]
            
            fig = px.bar(
                x=scores,
                y=books,
                orientation='h',
                title="Top 10 Recommendations",
                labels={'x': 'Similarity Score', 'y': 'Books'},
                color=scores,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                height=500,
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            st.subheader("🎯 Summary")
            avg_similarity = np.mean([rec['similarity'] for rec in st.session_state.recommendations])
            max_similarity = max([rec['similarity'] for rec in st.session_state.recommendations])
            
            st.metric("Average Similarity", f"{avg_similarity:.3f}")
            st.metric("Highest Similarity", f"{max_similarity:.3f}")
        
        else:
            # Instructions
            st.info("""
            📋 **Instructions:**
            
            1. Select a book from the sidebar dropdown
            2. Adjust the number of recommendations if needed
            3. Click 'Get Recommendations'
            4. Explore your personalized book suggestions!
            
            The similarity scores range from 0 to 1, where 1 means identical books.
            """)

else:
    st.error("Could not load the recommendation data. Please check your data files.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Made with ❤️ using Streamlit | Book Recommendation System v1.0
</div>
""", unsafe_allow_html=True)