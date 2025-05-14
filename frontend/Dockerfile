# ********************* Production Dockerfile ********************  
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Set build-time env var for API if needed (Vercel handles runtime via its UI)
# ARG NEXT_PUBLIC_API_BASE_URL 
# ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL

RUN npm run build

EXPOSE 3000
CMD ["npm", "start"] # Runs `next start`
