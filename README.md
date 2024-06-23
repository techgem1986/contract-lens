<h1>Contract Lens</h1>
<h2>1. Deploy models in Sagemaker</h2>

<li>Create a Sagemaker Notebook instance in AWS</li>
<li>Go to JupyterLab</li>
<li>Copy the notebook files from the sagemaker directory</li>
<li>Run the notebooks</li>

<p>This will create 2 sagemaker endpoints as 'minilm-embeddings' and 'mistral-llm' for embedding and predictions respectively</p>



<h2>2. Deploy the Python FastAPI app in EC2</h2>

<li>Launch an EC2 Instance in public subnet( Amazon Linux used in this example)</li>
<li>Connect to the EC2 instance</li>
<li>Run the below command to update the instance</li>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i>sudo yum update -y</i>
<li>Install the below packages</li>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i>sudo yum install -y git httpd python3-pip nginx</i>
<li>Configure nginx to add the below entry</li>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i>sudo vim /etc/nginx/nginx.conf</i>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;server {
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        listen 80;
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       server_name <EC2 Public IP addess>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       location / {
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          proxy_pass http://127.0.0.1:8000;
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      }
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   }
<li>Checkout the python code from github </li>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i>git clone https://github.com/techgem1986/contract-lens.git</i>
<li>Go to the directory /home/ec2-user/contract-lens/python</li>
<li>Install the required libraries</li>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i>pip install -r requirements.txt</i>
<li>Run aws configure to create the credential file</li>
<li>Start the python app</li>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i>python3 -m uvicorn app:app</i>


<h2>3. Deploy the UI in AWS Amplify</h2>

<li>TBU</li>