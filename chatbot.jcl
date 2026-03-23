 //CHATBOT   JOB (ACCT),'CHATBOT',CLASS=A,MSGCLASS=X             
 //SETVARS SET PYTHON='/u/usr/lpp/IBM/cyp/v3r13/pyz/bin/python'  
 //RUNIT  EXEC PGM=BPXBATCH,REGION=0M,                           
 //         PARM='SH &PYTHON /z/chatbot/app.py'      
 //STDOUT  DD SYSOUT=*                                           
 //STDERR  DD SYSOUT=*                                           