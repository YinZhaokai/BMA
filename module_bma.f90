      MODULE BMA
      IMPLICIT NONE

!
!        APRIL     2012  VEENHUIS  MDL CREATED
!
!        PURPOSE
!           A FORTRAN MODULE TO PERFORM BAYESIAN MODEL AVERAGING.
!           THE CODE IS BASED ON THE R LANGUAGE PACKAGE 
!           ENSEMBLEBMA BY FRALEY ET AL. 2011.  
!
!           THE TWO MAIN SUBROUTINES ARE FIT_BMA_NORMAL AND
!           BMA_FORECAST_CDF. FIT_BMA_NORMAL IS USED TO 
!           FIT THE BMA WEIGHTS. BMA_FORECAST_CDF 
!           IS USED TO PRODUCE A FORECAST CDF USING
!           THE BMA WEIGHTS AND PREDICTIVE ERROR VARIANCE.     
!
!
      CONTAINS
!
      SUBROUTINE FIT_BMA_NORMAL(ENSFORC,OBS,WEIGHTS,IGROUP,&
                                NOBS,NMEM,XMISS,XCONV,NTOT,&
                                XSIGMA,IEND,IER)
      IMPLICIT NONE
!
!        AUGUST    2011   VEENHUIS   MDL CREATED
!        AUGUST    2011   VEENHUIS   MADE MODS TO HANDLE MISSING DATA
!        SEPTEMBER 2011   VEENHUIS   ADDED XSIGMA TO CALL 
!        SEPTEMBER 2011   VEENHUIS   IF THE MEMBER IS NOT USED DUE TO
!                                    MISSING DATA SET THE FINAL WEIGHT
!                                    TO 0 RATHER THAN 9999. 
!        JANUARY   2012   VEENHUIS   IMPROVED DOCUMENTATION 
!        APRIL     2012   VEENHUIS   MODS TO MAKE THE CODE MORE GENERIC.
!                                    REPLACED HARDCODED 9999 VALUES WITH
!                                    XMISS. ADDED XMISS TO INPUT
!                                    CALL LIST. WEIGHTS IS SET TO
!                                    MISSING FOR A MEMBER WITH NO
!                                    GOOD DATA. 
!        APRIL     2012   VEENHUIS   DECIDED TO MOVE GROUP CHECKING CODE
!                                    INTO SUBROUINE TO PREVENT ERRORS
!                                    WHEN CALLING THE CODE. 
!        MAY       2012   VEENHUIS   FIXED A BUG WHEN ALL VALUES
!                                    OF A MEMBER WERE MISSING. 
!
!        PURPOSE
!           THIS SUBROUTINE USES BAYESIAN MODEL AVERAGING
!           TO FIT WEIGHTS AND PRODUCE A PREDICTIVE ERROR VARIANCE. 
!           THE CODE WAS ADAPTED FROM THE R LANGUAGE
!           FUNCTION FITBMANORMAL WHICH IS PART OF THE
!           PACKAGE ENSEMBLEBMA (FRALEY ET AL. 2011). THE CODE
!           FITS A GAUSSIAN MIXTURE MODEL USING THE EXPECTATION
!           MAXIMIZATION (EM) ALGORITHM.  
!                                             
!
!        VARIABLES
!
!             ENSFORC(I,J) = CONTAINS THE ENSEMBLE MEMBER FORECASTS
!                            WHERE I=1,NOBS AND J=1,NMEM.
!                            NOBS IS EQUAL TO THE LENGTH OF THE TRAINING
!                            SAMPLE. (INPUT)
!                   OBS(I) = CONTAINS THE VERIFYING OBSERVATIONS
!                            WHERE I=1,NOBS (INPUT).
!               WEIGHTS(J) = THE BMA WEIGHTS THAT WILL BE RETURNED
!                            WHERE J=1,NMEM (OUTPUT).
!                IGROUP(J) = ENSEMBLE MEMBER GROUPING INFORMATION.
!                            EACH ENTRY IS AN INTEGER VALUE.
!                            THIS IS USED TO GROUP ENSEMBLE MEMBERS
!                            INTO SUB ENSEMBLES THAT WILL BE CONSTRAINED
!                            TO HAVE EQUAL WEIGHTING.  WHERE J=1,NEM
!                            IF THE ENSMEMBLE MEMBERS
!                            ARE TO EACH RECIEVE THEIR OWN WEIGHT
!                            THEN IGROUP SHOULD CONTAIN A LIST OF
!                            INTEGERS FROM 1 TO NMEM.
!                            (INPUT).
!                     NOBS = THE NUMBER OF OBSERVATIONS USED FOR TRAINING.
!                            USED TO DIMENSION MULTIPLE ARRAYS. (INPUT).
!                     NMEM = THE NUMBER OF ENSEMBLE MEMBERS. (INPUT).
!                    XCONV = THE CONVERGENCE CRITERIA USED TO STOP
!                            INTERATING THE EM ALGORITHM. (INPUT).
!                     NTOT = MAXIMUM NUMBER OF ITERATIONS ALLOWED (INPUT).
!                   XSIGMA = THE BMA PREDICTED STANDARD DEVIATION (OUTPUT).
!                     IEND = THE REASON FOR STOPPING THE EM ALGORITHM.
!                            USEFUL FOR DIAGNOSTICS.
!                            0 = REACHED CONVERGENCE CRITERIA
!                            1 = REACHED MAXIMUM NUMBER OF ITERATIONS
!                                SPECIFIED BY NTOT. 
!                      IER = THE RETURN ERROR STATUS. (OUTPUT). 
!                   Z(I,J) = A MATRIX WITH DIMENSIONS I=1,NOBS J=1,NMEM.
!                            THE ELEMENTS OF Z ARE THE HEIGHT OF A NORMAL
!                            DISTRIBUTION WITH MEAN EQUAL TO THE MEMBER
!                            FORECAST AT THE OBSERVATION.  THIS ESSENTIALLY
!                            TELLS US HOW LIKELY THE OBSERVATION IS GIVEN
!                            THE ENSEMBLE MEMBER FORECAST. (INTERNAL).  
!                 ZSUM1(I) = FOR EACH OBSERVATION ZSUM1 CONTAINS THE SUMMATION
!                            OF THE ENSEMBLE MEMBER Z VALUES. WHERE I=1,NOBS
!                            (INTERNAL).
!                 ZSUM2(J) = FOR EACH MEMBER ZSUM2 CONTAINS THE SUMMATION
!                            OF THE Z VALUES FOR EACH OBSERVATION. 
!                            WHERE J=1,NMEM (INTERNAL).
!                 RSQ(I,J) = THE SQUARED ERROR OF EACH ENSEMBLE MEMBER
!                            FORECAST WHERE I=1,NOBS AND J=1,NMEM (INTERNAL). 
!               MATEX(J,I) = A MATRIX THAT CONTAINS ENSEMBLE GROUP MEMBERSHIP
!                            INFORMATION. J=1,NMEM AND I=1,NMEM (INTERNAL). 
!
!            
!
!       INPUT CALL VARIABLES
      INTEGER,INTENT(IN)  :: NOBS,NMEM,NTOT
      INTEGER,INTENT(IN)  :: IGROUP(NMEM)
      INTEGER,INTENT(OUT) :: IER,IEND
      REAL,INTENT(INOUT)  :: ENSFORC(NOBS,NMEM),OBS(NOBS)
      REAL,INTENT(IN)  :: XCONV
      REAL,INTENT(OUT) :: WEIGHTS(NMEM)
      REAL  XMISS,XSIGMA
!
!       LOCAL VARIABLES
      INTEGER :: I,J,JJ,M,MM,NN,IGLIST(NMEM)
      INTEGER :: NGROUPS,IFOUND
      INTEGER :: NVALID_OBS,NREQUIRED,NVALID_MEMBERS
      INTEGER :: NVALID_FORC,MEMBER_FLAG(NMEM)
      REAL :: Z(NOBS,NMEM),ZSUM1(NOBS),ZSUM2(NMEM)
      REAL :: RSQ(NOBS,NMEM),MATEX(NMEM,NMEM)
      REAL :: GROUP_COUNT,FACT,TERM1,TERM2
      REAL :: SD,LOGLIK,OLD,WCOUNT,WMEAN,WSUM,LTEST
!
      IER=0
      LOGLIK=0.
      OLD=0.
      XMISS=9999.
!
!
!        DETERMINE THE NUMBER OF UNIQUE ENSEMBLE GROUPINGS
!        BY EXAMINING THE ELEMENTS IF IGROUP
!
      NGROUPS=1
      IGLIST(1)=IGROUP(1)
!
      DO JJ=2,NMEM
!
        IFOUND=0
        DO MM=1,NGROUPS
          IF(IGROUP(JJ).EQ.IGLIST(MM))THEN
            IFOUND=1
          ENDIF
        ENDDO
!
        IF(IFOUND.EQ.0)THEN
          NGROUPS=NGROUPS+1
          IGLIST(NGROUPS)=IGROUP(JJ)
        ENDIF
!
      ENDDO 
!
!
!        DO SOME INTIIAL CHECKS FOR MISSING DATA
!
!
!
      NVALID_OBS=0
      DO J=1,NOBS
        IF(OBS(J).NE.XMISS)THEN
          NVALID_OBS=NVALID_OBS+1
        ENDIF
      ENDDO
!
!        IF THERE ARE NO VALID  OBS THEN ABORT.
!
      IF(NVALID_OBS.LE.0) THEN
        WRITE(*,*) " IN SUBROUTINE FIT_BMA_NORMAL NO VALID ",&
                   "OBSERVATIONS FOUND"
        WRITE(*,*) " CANNOT PERFORM BMA."
        IER=98
        RETURN
      ENDIF 
!
!        FOR EACH MEMBER COUNT THE NUMBER OF VALID
!        FORECASTS.  IF THERE ARE NO GOOD FORECASTS
!        FOR MEMBER I THEN MEMBER_FLAG(I) WILL BE SET 
!        TO 0.  OTHERWISE MEMBER_FLAG(I) WILL BE
!        SET TO 1.
!
      NVALID_MEMBERS=0
      DO I=1,NMEM
!
!        COUNT THE NUMBER OF VALID
!        FORECASTS FOR THIS MEMBER
!
        NVALID_FORC=0 
        DO J=1,NOBS
          IF(ENSFORC(J,I).NE.XMISS)THEN
            NVALID_FORC=NVALID_FORC+1
          ENDIF
        ENDDO
!
        IF(NVALID_FORC.LE.0) THEN
          ENSFORC(1:NOBS,I)=XMISS
          MEMBER_FLAG(I)=0
        ELSE 
          NVALID_MEMBERS=NVALID_MEMBERS+1
          MEMBER_FLAG(I)=1
        ENDIF
!
      ENDDO
!
!        REQUIRE AT LEAST TWO GOOD ENSEMBLE MEMBERS TO DO
!        BMA.
!
      IF(NVALID_MEMBERS.LT.2) THEN
        WRITE(*,*) " IN SUBROUTINE FIT_BMA_NORMAL"
        WRITE(*,*) " LESS THAN 2 VALID ENSEMBLE MEMBERS"
        WRITE(*,*) " CANNOT PERFORM BMA"
        IER=99
        RETURN
      ENDIF
! 
!        SET THE INITIAL WEIGHTS
!
      WEIGHTS(1:NMEM)=XMISS
      DO I=1,NMEM
        IF(MEMBER_FLAG(I).EQ.1)THEN
          WEIGHTS(I)=1./(1.*NVALID_MEMBERS)
        ENDIF
      ENDDO
!
!
!        CALCULATE THE INITIAL VALUE OF SD.
!        SD IS THE STANDARD DEVIATION.
!
!
      SD=STDDEV(OBS,NOBS,XMISS) 
!
      DO J=1,NOBS
      DO I=1,NMEM
        IF((OBS(J).NE.XMISS).AND.(ENSFORC(J,I).NE.XMISS))THEN
          RSQ(J,I)=(OBS(J)-ENSFORC(J,I))**2.
        ELSE
          RSQ(J,I)=XMISS
        ENDIF
      ENDDO
      ENDDO
!
!        SET UP THE MATRIX MATEX. MATEX IS USED TO UPDATE THE
!        ESTIMATE OF THE STANDARD DEVIATION.
!
      MATEX(:,:)=0.
      DO NN=1,NGROUPS
        GROUP_COUNT=0.
        DO MM=1,NMEM
          IF((IGROUP(MM).EQ.IGLIST(NN)).AND.&
              (MEMBER_FLAG(I).EQ.1)) THEN
            GROUP_COUNT=GROUP_COUNT+1.
          ENDIF
        ENDDO
!
        IF(GROUP_COUNT .GT. 0) THEN
          FACT=1./GROUP_COUNT
!        ELSE IF(GROUP_COUNT.EQ.1) THEN
!          FACT=1.
        ELSE
          FACT=1.
        ENDIF 
!
        DO MM=1,NMEM
          IF(IGROUP(MM).EQ.IGLIST(NN)) THEN
            MATEX(MM,NN)=FACT
          ENDIF
        ENDDO
      ENDDO
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  
!        BEGIN ITERATING THE EM ALGORITHM
!
!        M AND LTEST ARE USED TO CONTROL THE WHILE LOOP.
!        THE MAX NUMBER OF ITERATIONS IS LIMITED TO 50 TO
!        AVOID AN INFINITE LOOP.  THE VALUE OF LTEST CONTAINS
!        THE CHANGE IN LIKELIHOOD FUNCTION AFTER EACH ITERATION.
!        XCONV IS THE STOPPING CRITERIA.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
      M=1
      LTEST=1.
      DO WHILE((M .LT. NTOT).AND.(LTEST .GT. XCONV))
!
!        CALC THE MATRIX Z
!
      Z(1:NOBS,1:NMEM)=XMISS
      DO J=1,NOBS
      DO I=1,NMEM
        IF((OBS(J).NE.XMISS).AND.(ENSFORC(J,I).NE.XMISS))THEN
          Z(J,I)=DNORM(OBS(J),ENSFORC(J,I),SD)
        ELSE
          Z(J,I)=XMISS
        ENDIF
      ENDDO
      ENDDO
!
!        MULTIPLY BY THE WEIGHTS
!
      DO J=1,NOBS
      DO I=1,NMEM
        IF((Z(J,I).NE.XMISS).AND.(WEIGHTS(I).NE.XMISS))THEN 
          Z(J,I)=Z(J,I)*WEIGHTS(I)
        ENDIF
      ENDDO
      ENDDO
!
!
      ZSUM1(1:NOBS)=0.
      DO J=1,NOBS
        ZSUM1(J)=SUM(Z(J,1:NMEM),MASK=Z(J,1:NMEM).NE.XMISS)
!
!          IF ALL THE MEMBERS WERE MISSING THEN MAKE SURE
!          ZSUM1 IS SET TO MISSING 
!
        IF(ZSUM1(J).EQ.0) THEN
          ZSUM1(J)=XMISS
        ENDIF 
!
        DO I=1,NMEM
          IF((Z(J,I).NE.XMISS).AND.(ZSUM1(J).NE.XMISS)) THEN
             Z(J,I)=Z(J,I)/ZSUM1(J)
          ELSE
            Z(J,I)=XMISS
          ENDIF   
        ENDDO
      ENDDO
!
!        UPDATE LOG LIKELIHOOD FUNCTION
!
      OLD=LOGLIK
      LOGLIK=SUM(LOG(ZSUM1),MASK = ZSUM1 .NE. XMISS) 
!
      ZSUM2(1:NMEM)=0.
      DO I=1,NMEM
       IF(MEMBER_FLAG(I).EQ.1) THEN
        ZSUM2(I)=SUM(Z(1:NOBS,I),MASK = Z(1:NOBS,I) .NE. XMISS)
       ENDIF
      ENDDO
!
!        CALCULATE THE NEW ESTIMATE OF THE WEIGHTS
!
      DO I=1,NMEM
      IF((SUM(ZSUM2).GT.0).AND.(MEMBER_FLAG(I).EQ.1))THEN
        WEIGHTS(I)=ZSUM2(I)/SUM(ZSUM2)
      ELSE
        WEIGHTS(I)=XMISS
      ENDIF
      ENDDO
!
!
      IF(M.GT.1) THEN
        LTEST=(LOGLIK-OLD)/(1.+ABS(LOGLIK))
      ENDIF 
!
!        UPDATE THE STANDARD DEVIATION
!
      IF(NGROUPS .EQ. NMEM) THEN
!        
!          IF EACH MEMBER IS ALLOWED ITS OWN WEIGHT THEN
!          THE UPDATE IS SIMPLE 
!        
!
        SD=(SUM(Z*RSQ,MASK=Z.NE.XMISS)/SUM(Z,MASK=Z.NE.XMISS))**(.5)
!
      ELSE
!        
!          OTHERWISE THE UPDATE IS MORE COMPLICATED.
!          AVERAGE THE WEIGHTS IN EACH GROUP
!
        DO NN=1,NGROUPS
          WCOUNT=0.
          WSUM=0.
          WMEAN=0.
!
          DO MM=1,NMEM
            IF((IGROUP(MM).EQ.IGLIST(NN)).AND.&
               (MEMBER_FLAG(MM).EQ.1)) THEN
              WCOUNT=WCOUNT+1.
              WSUM=WSUM+WEIGHTS(MM)
            ENDIF
          ENDDO
!
          IF(WCOUNT.GT.0) THEN
            WMEAN=WSUM/WCOUNT
          ENDIF
!
          DO MM=1,NMEM 
            IF((IGROUP(MM).EQ.IGLIST(NN)).AND.&
               (MEMBER_FLAG(MM).EQ.1)) THEN
              WEIGHTS(MM)=WMEAN
            ENDIF
          ENDDO
        ENDDO
! 
! 100    FORMAT(/,I8,X,3(F5.4,X))
!
!          UPDATE THE ESTIMATE OF SD
!
!          FLIP THE MISSING VALUES TO 0 TO GET
!          THE MATRIX MULTIPLICATION TO WORK
!
        DO J=1,NOBS
        DO I=1,NMEM
          IF((MEMBER_FLAG(I).EQ.1).AND.&
             (Z(J,I).EQ.XMISS))THEN 
            Z(J,1:NMEM)=0.
            RSQ(J,1:NMEM)=0.
          ELSE IF(MEMBER_FLAG(I).EQ.0) THEN
            Z(J,I)=0.
            RSQ(J,I)=0.
          ENDIF 
        ENDDO   
        ENDDO
!
! 
        TERM1=SUM(MATMUL((Z*RSQ),MATEX(1:NMEM,1:NGROUPS)))
        TERM2=SUM(MATMUL((Z),MATEX(1:NMEM,1:NGROUPS)))
        SD=(TERM1/TERM2)**(.5)
!
      ENDIF
!
      M=M+1
!
!        END OF DO WHILE LOOP
      ENDDO
!
!        FIGURE OUT WHICH STOPPING CRITERIA WAS MET
!
      IF(M .GE. NTOT) THEN
        IEND=1
      ELSE
        IEND=0
      ENDIF
!
!        WE HAVE EXITED THE EM
!        COPY THE STDDEV TO XSIGMA
!
      XSIGMA=SD
!
      END SUBROUTINE
!
!
!
      REAL FUNCTION DNORM(X,MU,SD)
      IMPLICIT NONE
!
!        APRIL     2012   VEENHUIS   MDL - CREATED
!
!        PURPOSE
!           COMPUTES THE VALUE OF A NORMAL DISTRIBUTION
!           AT A POINT.
!
!        VARIABLES
!                        X = THE LOCATION TO COMPUTE (INPUT). 
!                       MU = THE MEAN PARAMETER FOR THE NORMAL FUNCTION (INPUT)
!                       SD = THE STANDARD DEVIATION OF THE NORMAL FUNCTION (INPUT)
!        INPUT VARIABLES
!
      REAL,INTENT(IN) :: X,MU,SD
!
!        LOCAL VARIABLES
      REAL :: PI=3.14159,TERM1,TERM2,RESULT
!
      TERM1=1./(2*PI*(SD**2.))**(.5)
      TERM2=(-(X-MU)**2.)/(2*(SD**2.))
      RESULT=TERM1*EXP(TERM2)
!
      DNORM=RESULT
      RETURN
      END FUNCTION
!
!
      REAL FUNCTION STDDEV(X,N,XMISS)
      IMPLICIT NONE
!
!        APRIL     2012   VEENHUIS   MDL - CREATED
!
!        PURPOSE
!           COMPUTES THE STANDARD DEVIATION OF AN ARRAY. 
!
!        VARIABLES
!                     X(I) = THE ARRAY OF VALUES WHERE I=1,N (INPUT). 
!                        N = THE LENGTH OF THE INPUT ARRAY X (INPUT).
!                    XMISS = THE VALUE TO USE FOR MISSING VALUES.
!                            MISSING VALUES WILL BE IGNORED. (INPUT).
!        INPUT VARIABLES
      INTEGER,INTENT(IN) :: N
      REAL,INTENT(IN) :: X(N),XMISS
!
!        LOCAL VARIABLES
      REAL :: MEAN,XSUM2,XSUM
      INTEGER :: I,NTRUE
! 
      XSUM=0.
      NTRUE=0
      DO I=1,N
        IF(X(I).NE.XMISS) THEN
          XSUM=XSUM+X(I)
          NTRUE=NTRUE+1
        ENDIF
      ENDDO
!
      MEAN=XSUM/(1.*NTRUE)
!
      XSUM2=0.
      DO I=1,N
        IF(X(I).NE.XMISS) THEN
          XSUM2=XSUM2+(X(I)-MEAN)**2.
        ENDIF
      ENDDO
!
      STDDEV=((XSUM2)/(1.*NTRUE-1))**(.5)
      RETURN
      END FUNCTION
!
!
!
      SUBROUTINE BMA_FORECAST_CDF(CDF_POINTS,FCST,XSIGMA,&
                                  WEIGHTS_IN,XVALUES,XMISS,&
                                  NMEM,NCDF,IER) 
      IMPLICIT NONE
!        APRIL     2012   VEENHUIS   MDL - CREATED
!
!        PURPOSE
!           RETURNS THE VALUES OF A CDF AT MULTIPLE PERCENT LEVELS.
!           THE INPUTS ARE THE BMA WEIGHTS, BMA PREDICTIVE 
!           STANDARD ERROR AND THE DESIRED CDF POINTS. THE CODE 
!           RETURNS THE CDF VALUES IN THE ARRAY XVALUES. A CALL
!           SHOULD BE MADE TO FIT_BMA_NORMAL PRIOR TO CALLING
!           THIS SUBROUTINE.
!
!        VARIABLES
!            CDF_POINTS(I) = THE CDF PERCENTILE VALUES FOR WHICH
!                            FORECASTS ARE DESIRED. WHERE I=1,NCDF
!                            (INPUT). 
!                  FCST(J) = THE ENSEMBLE MEMBER FORECASTS. WHERE J=1,NMEM
!                            (INPUT). 
!                   XSIGMA = THE PREDICTIVE ERROR VARIANCE OBTAINED BY
!                            FITTING THE BMA WEIGHTS. (INPUT)
!            WEIGHTS_IN(J) = THE WEIGHTS FOR EACH MEMBER J WHERE
!                            J=1,NMEM (INPUT)
!               XVALUES(I) = THE COMPUTED CDF FORECASTS WHERE I=1,NCDF.
!                            (OUTPUT).
!                    XMISS = THE VALUE USED TO FLAG MISSING DATA (INPUT).
!                     NMEM = THE NUMBER OF ENSEMBLE MEMBERS (INPUT).
!                     NCDF = THE NUMBER OF CDF POINTS TO COMPUTE (INPUT).
!                      IER = THE ERROR RETURN STATUS (OUTPUT).
!                   NVALID = THE NUMBER OF VALID MEMBER FORECAST (INTERNAL).
!                IFLAGS(J) = AN ARRAY USE TO FLAG IF A MEMBER SHOULD BE USED
!                            WHERE J=1,NMEM (INTERNAL).
!                            0 = DO NOT USE THIS MEMBER. EITHER THE WEIGHT
!                                OR FORECAST IS MISSING.
!                            1 = OK TO USE THIS MEMBER
!               WEIGHTS(J) = LOCAL COPY OF THE WEIGHTS FOR 
!                            EACH MEMBER J WHERE J=1,NMEM (INTERNAL).
!                     WSUM = SUM OF THE WEIGHTS. USED TO CHECK THAT WEIGHTS
!                            SUM TO UNITY. (INTERNAL).
!
!        INPUT VARIABLES
      INTEGER,INTENT(IN)  :: NMEM,NCDF
      INTEGER,INTENT(OUT) :: IER
      REAL,INTENT(IN)   :: CDF_POINTS(NCDF)
      REAL,INTENT(IN)   :: FCST(NMEM),XSIGMA,WEIGHTS_IN(NMEM)
      REAL,INTENT(OUT)  :: XVALUES(NCDF)
      real XMISS
!
!        LOCAL VARIABLES
      INTEGER :: I,NVALID
      INTEGER :: IFLAGS(NMEM)
      REAL :: XX,WEIGHTS(NMEM),SIGMA(NMEM),WSUM
!
      WSUM=0.
      IFLAGS(1:NMEM)=0
      WEIGHTS(1:NMEM)=XMISS
      DO I=1,NMEM
       IF((WEIGHTS_IN(I).NE.XMISS).AND.(FCST(I).NE.XMISS))THEN
          IFLAGS(I)=1
          WEIGHTS(I)=WEIGHTS_IN(I)
          SIGMA(I)=XSIGMA
          WSUM=WSUM+WEIGHTS_IN(I)
          NVALID=NVALID+1
!       ELSE IF(FCST(I).EQ.XMISS)THEN
!         WRITE(*,*) "WARNING IN SUBROUTINE BMA_FORECAST_CDF"
!         WRITE(*,*) "THE FORECAST FOR MEMBER",I," IS MISSING"
!         WRITE(*,*) "THE WEIGHTS WILL BE RESCALED"
       ENDIF
      ENDDO
!
      IF(NVALID.NE.NMEM)THEN
        DO I=1,NMEM
          IF(IFLAGS(I).EQ.1) THEN
            WEIGHTS(I)=WEIGHTS(I)/WSUM
          ENDIF
        ENDDO
      ENDIF
!
      XVALUES(1:NCDF)=-9999.
!
!       SINCE THIS IS A PUBLIC FACING SUBROUTINE DO SOME
!       EXTENSIVE ERROR CHECKING
!
!       MAKE SURE CDF POINTS ARE VALID
!
      IF((MAXVAL(CDF_POINTS(1:NCDF)).GT.1).OR.&
         (MINVAL(CDF_POINTS(1:NCDF)).LT.0))THEN
        WRITE(*,*) "ERROR IN SUBROUTINE BMA_FORECAST_CDF" 
        WRITE(*,*) "INVALID VALUES IN CDF_POINTS" 
        IER=97
        RETURN
      ENDIF
!
!       CHECK FOR NEGATIVE WEIGHTS
!
      IF(MINVAL(WEIGHTS(1:NMEM),MASK=IFLAGS.EQ.1).LT.0) THEN
        WRITE(*,*) "ERROR IN SUBROUTINE BMA_FORECAST_CDF" 
        WRITE(*,*) "ENCOUNTERED NEGATIVE WEIGHTS"
        WRITE(*,*) "RETURNING..."
        IER=96
        RETURN
      ENDIF
!
!       CHECK FOR WEIGHTS GREATER THAN 1
!
      IF(MAXVAL(WEIGHTS(1:NMEM),MASK=IFLAGS.EQ.1).GT.1) THEN
        WRITE(*,*) "ERROR IN SUBROUTINE BMA_FORECAST_CDF" 
        WRITE(*,*) "ENCOUNTERED WEIGHTS GREATER THAN 1"
        WRITE(*,*) "RETURNING..."
        IER=95
        RETURN
      ENDIF
!
!       MAKE SURE WEIGHTS SUM TO UNITY
!
      IF(ABS(1-SUM(WEIGHTS(1:NMEM),MASK= IFLAGS.EQ.1))&
         .GT. 0.001)THEN
!        WRITE(*,*) "ERROR IN SUBROUTINE BMA_FORECAST_CDF"
!        WRITE(*,*) "WEIGHTS DO NOT SUM TO UNITY"
!        WRITE(*,*) "RETURNING..."
        IER=94
        RETURN
      ENDIF
!
!       CHECK FOR NEGATIVE SIGMA VALUES 
!
      IF(MINVAL(SIGMA(1:NMEM),MASK=IFLAGS.EQ.1).LT.0) THEN
        WRITE(*,*) "ERROR IN SUBROUTINE BMA_FORECAST_CDF"
        WRITE(*,*) "ENCOUNTERED NEGATIVE STANDARD DEVIATIONS"
        WRITE(*,*) "RETURNING..."
        IER=93
        RETURN
      ENDIF
!
!
      DO I=1,NCDF
!
        IF((CDF_POINTS(I).GE.0).AND.(CDF_POINTS(I).LE.1))THEN
!
!           THE FUNCTION BISECT USES A ROOT FINDING ALGORITHM
!           TO COMPUTE THE VALUES OF THE CDF.
!
          CALL BISECT(CDF_POINTS(I),FCST,SIGMA,WEIGHTS,IFLAGS,&
                                    NMEM,XX,IER)
          IF(IER.EQ.0)THEN
            XVALUES(I)=XX
          ELSE
            XVALUES(I)=-9999.
            IER=92
          ENDIF
        ELSE
          XVALUES(I)=-9999.
        ENDIF
!
      ENDDO
!
      END SUBROUTINE 
!
!
      SUBROUTINE BISECT(CDF_POINT,FCST,SIGMA,WEIGHTS,IFLAGS,&
                              NFCST,XVALUE,IER)
      IMPLICIT NONE
!
!        APRIL     2012   VEENHUIS   MDL - CREATED
!
!        PURPOSE
!           USES THE BISECTION METHOD TO FIND POINTS ALONG
!           A CDF. 
!
!        VARIABLES
!                CDF_POINT = THE PERCENTILE VALUE OF THE
!                            CDF POINT DESIRED (INPUT).
!                  FCST(J) = THE ENSEMBLE MEMBER FORECASTS WHERE
!                            J=1,NFCST (INPUT).
!                 SIGMA(J) = THE STANDARD DEVIATION VALUES TO 
!                            ATTACH TO EACH ENSEMBLE MEMBER WHERE
!                            J=1,NFCST (INTERNAL).
!               WEIGHTS(J) = THE BMA WEIGHTS FOR EACH MEMBER WHERE
!                            J=1,NFCST (INTERNAL).
!                IFLAGS(J) = AN ARRAY USE TO FLAG IF A MEMBER SHOULD BE USED
!                            (INTERNAL).
!                            0 = DO NOT USE THIS MEMBER. EITHER THE WEIGHT
!                                OR FORECAST IS MISSING.
!                            1 = OK TO USE THIS MEMBER
!                    NFCST = THE NUMBER OF ENSEMBLE MEMBERS (INPUT).
!                   XVALUE = THE RETURNED VALUE OF THE CDF (OUTPUT).
!                      IER = THE ERROR STATUS (OUTPUT).
!                    LOWER = USED BY THE BISECTION ALGORITHM (INTERNAL).
!                    UPPER = USED BY THE BISECTION ALGORITHM (INTERNAL).
!                   YLOWER = USED BY THE BISECTION ALGORITHM (INTERNAL).
!                   YUPPER = USED BY THE BISECTION ALGORITHM (INTERNAL).
!                    XDIFF = USED BY THE BISECTION ALGORITHM (INTERNAL).
!                    YHALF = USED BY THE BISECTION ALGORITHM (INTERNAL).
!                     HALF = USED BY THE BISECTION ALGORITHM (INTERNAL).
!                   XCOUNT = NUMBER OF ITERNATIONS OF THE BISECTION 
!                            ALGORITHM. USED TO AVOID INFITE LOOPS. (INTERNAL). 
!                MAX_SIGMA = USED BY THE BISECTION ALGORITHM
!                            TO DETERMINE STARTING POINTS (INTERNAL).
!                      TOL = STOPPING TOLERANCE FOR THE ALGORITHM (INTERNAL).
!           NVALID_MEMBERS = NUMBER OF VALID ENSEMBLE MEMBERS
!                            (INTERNAL).
!        INPUT VARIBLES     
      INTEGER,INTENT(IN)  :: NFCST
      INTEGER,INTENT(IN)  :: IFLAGS(NFCST) 
      INTEGER,INTENT(OUT) :: IER 
      REAL,INTENT(IN)     :: CDF_POINT
      REAL,INTENT(IN)     :: FCST(NFCST),SIGMA(NFCST),WEIGHTS(NFCST)
      REAL,INTENT(OUT)    :: XVALUE 
!
!        LOCAL VARIABLES
      INTEGER :: NVALID_MEMBERS
      REAL    :: LOWER,UPPER,MAX_SIGMA,YLOWER,YUPPER
      REAL    :: TOL=.0001,XDIFF,YHALF,HALF,XCOUNT
!
      IER=-1
!
!        COUNT THE NUMBER OF VALID MEMBERS
!
      NVALID_MEMBERS=SUM(IFLAGS(1:NFCST))
      IF(NVALID_MEMBERS .GT. 0) THEN
!
!         STARTING POINTS FOR THE BISECTION ALGORITHM
!
        MAX_SIGMA=MAXVAL(SIGMA(1:NFCST),MASK=IFLAGS.EQ.1)      
        LOWER=MINVAL(FCST(1:NFCST),MASK=IFLAGS.EQ.1)
        UPPER=MAXVAL(FCST(1:NFCST),MASK=IFLAGS.EQ.1)
!
        LOWER=LOWER-3.*MAX_SIGMA
        UPPER=UPPER+3.*MAX_SIGMA

!      
      ELSE
!
        WRITE(*,*) "IN BISECT NO VALID MEMBERS TO PROCESS"
        WRITE(*,*) "RETURNING..."
!
        XVALUE=-9999
        IER=-1
        RETURN
!
      ENDIF 
!
!        COMPUTE THE XVALUES FOR THE LOWER AND UPPER STARTING
!        POINTS.
!
      CALL CDF_NORMAL_MIXTURE(LOWER,FCST,SIGMA,WEIGHTS,IFLAGS,&
                              NFCST,YLOWER,IER) 
      CALL CDF_NORMAL_MIXTURE(UPPER,FCST,SIGMA,WEIGHTS,IFLAGS,&
                              NFCST,YUPPER,IER) 
!
!        CHECK IF THE POINT WE DESIRE IS OUTSIDE THE RANGE.
!
      XVALUE=-9999.
      IF(CDF_POINT .LE. YLOWER) THEN
        XVALUE=0.0
        IER=0
      ELSE IF (CDF_POINT .GE. YUPPER) THEN
        XVALUE=1.0
        IER=0
      ELSE
!
!          BEGIN ITERATING THE SEARCH. THE TOTAL NUMBER
!          OF ITERATIONS IS LIMITED TO 20 TO PREVENT AN 
!          INFINITE LOOP CONDITION.
!
        XDIFF=1.
        XCOUNT=0.
        DO WHILE((ABS(XDIFF).GT.TOL).AND.(XCOUNT.LT.20))
!    
          HALF=(LOWER+UPPER)/2.
          CALL CDF_NORMAL_MIXTURE(HALF,FCST,SIGMA,&
                  WEIGHTS,IFLAGS,NFCST,YHALF,IER)
!
          XDIFF=ABS(YHALF-CDF_POINT)         
          IF(XDIFF.LE.TOL) THEN
            XVALUE=HALF
            IER=0
          ELSE
            IF(CDF_POINT .LE. YHALF)THEN
              UPPER=HALF
            ELSE
              LOWER=HALF
            ENDIF
          ENDIF
!
          XCOUNT=XCOUNT+1.
          IF(XCOUNT .GE. 20) IER=-1
        ENDDO
!
      ENDIF
!
      END SUBROUTINE 
!
!
      SUBROUTINE CDF_NORMAL_MIXTURE(X,FCST,SIGMA,WEIGHTS,IFLAGS,&
                                    NFCST,Y,IER)
      IMPLICIT NONE
!
!        APRIL     2012   VEENHUIS   MDL - CREATED
!
!        PURPOSE
!           COMPUTES THE CDF OF A NORMAL MIXTURE. 
!
!        VARIABLES
!                        X = PERCENTILE POINT OF THE CDF THAT 
!                            IS DESIRED (INPUT).
!                  FCST(J) = THE ENSEMBLE MEMBER FORECASTS WHERE
!                            J=1,NFCST (INPUT).
!                 SIGMA(J) = THE STANDARD DEVIATION VALUES FOR 
!                            EACH NORMAL COMPONENT WHERE J=1,NFCST
!                            (INPUT).
!               WEIGHTS(J) = THE WEIGHTS FOR EACH MEMBER WHERE J=1,
!                             NFCST (INPUT). 
!                IFLAGS(J) = AN ARRAY USED TO FLAG IF A MEMBER SHOULD BE USED
!                            (INTERNAL).
!                            0 = DO NOT USE THIS MEMBER. EITHER THE WEIGHT
!                                OR FORECAST IS MISSING.
!                            1 = OK TO USE THIS MEMBER
!                    NFCST = THE NUMBER OF ENSEMBLE MEMBERS (INPUT).
!                        Y = THE RETURNED VALUES OF THE CDF (OUTPUT).
!                      IER = THE ERROR RETURN STATUS (OUTPUT).
!
!
!        INPUT VARIABLES 
      INTEGER,INTENT(IN)  :: NFCST 
      INTEGER,INTENT(IN)  :: IFLAGS(NFCST)
      INTEGER,INTENT(OUT) :: IER 
      REAL,INTENT(IN)     :: X,FCST(NFCST),SIGMA(NFCST),WEIGHTS(NFCST)
      REAL,INTENT(OUT)    :: Y
!
!        LOCAL VARIABLES
      REAL :: Z
      INTEGER :: I
!
!        MAKE SURE WEIGHTS SUM TO UNITY
!
      IF(ABS(1-SUM(WEIGHTS(1:NFCST),MASK= IFLAGS.EQ.1))&
         .GT. 0.0001)THEN
!        WRITE(*,*) "ERROR IN SUBROUTINE CDF_NORMAL_MIXTURE"
!        WRITE(*,*) "WEIGHTS DO NOT SUM TO UNITY"
!        WRITE(*,*) "RETURNING..."
        IER=-1
        RETURN
      ENDIF
!
      IF(MINVAL(SIGMA(1:NFCST),MASK=IFLAGS.EQ.1).LE.0) THEN
        WRITE(*,*) "ERROR IN SUBROUTINE CDF_NORMAL_MIXTURE"
        WRITE(*,*) "TRIED TO PROCESS A STANDARD DEVIATION"
        WRITE(*,*) "LESS THAN OR EQUAL TO 0"
        WRITE(*,*) "RETURNING..."
        IER=-1
        RETURN
      ENDIF
!
!        THE CHECKS ABOVE WERE PASSED SO COMPUTE THE CDF
!
      Y=0.
      DO I=1,NFCST
        IF(IFLAGS(I).EQ.1)THEN
          CALL CNORM(X,FCST(I),SIGMA(I),Z)
          Y=Y+Z*WEIGHTS(I)
        ENDIF
      ENDDO
!
      END SUBROUTINE
!
!
      SUBROUTINE CNORM(X,MEAN,SIGMA,Z) 
      IMPLICIT NONE
!
!        APRIL     2012   VEENHUIS   MDL - CREATED
!
!        PURPOSE
!           COMPUTES THE CDF OF A SINGLE NORMAL DENSITY.
!           CODE USES THE INTRINSIC FUNCTION ERF WHICH
!           IS THE ERROR FUNCTION. 
!
!        VARIABLES
!                        X = PERCENTILE POINT OF THE CDF THAT 
!                            IS DESIRED (INPUT).
!                     MEAN = MEAN OF THE NORMAL DISTRIBUTION (INPUT).
!                    SIGMA = STANDARD DEVIATION OF THE NORMAL 
!                            DISTRIBUTION (INPUT).
!                        Z = RETURNED CDF VALUE (OUTPUT).
!
!        INPUT VARIABLES
      REAL,INTENT(IN)  :: X,MEAN,SIGMA
      REAL,INTENT(OUT) :: Z 
!
      Z=0.5*(1+ERF((X-MEAN)/SQRT(2.*(SIGMA**2.)))) 
!
      RETURN
      END SUBROUTINE 
!
      END MODULE 
