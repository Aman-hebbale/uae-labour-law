FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy frontend code
COPY . .

# 4. Install Streamlit
RUN pip install streamlit requests

# 5. Run Streamlit on the port Google Cloud expects
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0