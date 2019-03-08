!     EXAMPLE PROGRAM OF HOW TO USE THE BMA MODULE.
!     FOR THIS EXAMPLE THE ENSEMBLE FORECASTS 
!     AND CORRESPONDING VERIFYING
!     OBSERVATIONS ARE READ FROM AN ASCII FILE.
!
!     VEENHUIS
!
!     IT IS THE USERS JOB TO CORRECTLY POPULATE THE ARRAYS
!     WITH THE TRAINING DATA.
!
!
!     CONSTANT VARIABLES
!
!                   NCDF = NUMBER OF POINTS DESIRED TO SAVE ALONG
!                          THE CDF.
!                   NMEM = NUMBER OF ENSEMBLE MEMBERS.
!                 NLINES = NUMBER OF LINES IN THE ASCII INPUT FILE. 
!                 NTRAIN = LENGTH OF TRAINING SAMPLE TO USE.
!                NGROUPS = NUMBER OF SUB ENSEMBLE GROUPINGS. IF EACH
!                          MEMBER IS TO RECEIVE ITS OWN UNIQUE WEIGHT
!                          THEN NGROUPS SHOULD EQUAL NMEM. 
!             PROJECTION = THE FORECAST PROJECTION IN HOURS.
!
!
!     DUMMY VARIABLES
!
!                STATION = CHARACTER VARIABLE USE TO READ STATION
!                          FROM ASCII FILE.
!                   TOBS = USED TO HOLD OBSERVATIONS WHEN READING FILE.
!      TMEM1,TMEM2,TMEM3 = USE TO HOLD THE MEMBER FORECASTS
!                          WHEN READING THE FILE. 
!                  TDATE = USE TO HOLD THE DATE WHEN READING THE FILE.
!
!     VARIABLES
!
!      FCST(NLINES,NMEM) = HOLDS ENSEMBLE MEMBER FORECAST READ FROM
!                          FILE. 
!            OBS(NLINES) = HOLDS VERIFYING OBSERVATIONS.   
!           DATE(NLINES) = HOLDS THE DATES.   
!          WEIGHTS(NMEM) = BMA WEIGHTS RETURNED BY FIT_BMA_NORMAL. 
!           IFLAGS(NMEM) = SET BY FIT_BMA_NORMAL. USE TO INDICATE IF
!                          THE CODE WAS ABLE TO FIT A WEIGHT TO EACH
!                          MEMBER. 
!                          0 = INDICATES A WEIGHT COULD NOT BE FIT 
!                          1 = INDICATES A WEIGHT WAS FIT.  
!                    IER = ERROR STATUS FROM SUBROUTINE CALL.
!           IGROUP(NMEM) = HOLDS ENSEMBLE GROUPING INFORMATION. IF CERTAIN MEMBERS
!                          ARE TO BE CONSTRAINED TO HAVE EQUAL WEIGHTS, THEN
!                          VALUES FOR IGROUP SHOULD BE THE SAME. 
!                  XMISS = THE VALUE TO USE TO INDICATE MISSING VALUES.
!                  XCONV = THE CONVERGENCE CRITERIA USED TO STOP THE BMA
!                          ALGORITHM.  MEASURE THE SUCCESSIVE CHANGE IN THE 
!                          LOG LIKELIHOOD FUNCTION.
!                   NTOT = TOTAL NUMBER OF ITERATIONS ALLOWED TO FIT THE WEIGHTS. 
!                 XSIGMA = THE PREDICTIVE STANDARD ERROR RETURNED BY FIT_BMA_NORMAL.
!                   IEND = INFORMATION CONCERNING WHICH STOPPING CRITIERIA
!                          WAS MET.
!                          0 = CONVERGENCE CRITERIA XCONV WAS MET
!                          1 = TOTAL NUMBER OF ITERATIONS GREATER THAN NTOT
!       CDF_POINTS(NCDF) = THE CDF POINTS DESIRED BY THE USERS.
!          XVALUES(NCDF) = THE CORRESPONDING VALUES OF THE CDF POINTS IN THE 
!                          ARRAY CDF_POINTS. RETURNED BY ????.
!
      PROGRAM MAIN 
      USE BMA 
      IMPLICIT NONE
!
!       CONSTANTS USED TO DIMENSION ARRAYS
!
      INTEGER,PARAMETER   :: NCDF=5
      INTEGER :: NTRAIN, NLINES, NMEM, NMEM1, NMEM2
      REAL,PARAMETER      :: PROJECTION=102.
!
      CHARACTER(LEN=100)        :: FCST_FILE, OBS_FILE
      CHARACTER(LEN=3)        :: NTRAIN_STR, NLINES_STR, NMEM_STR1, NMEM_STR2

!       DUMMY VARIABLES USED TO READ DATA FROM FILE
!
      INTEGER :: TIME_LABEL, TDATE
      REAL :: TOBS
      REAL,DIMENSION(:), ALLOCATABLE :: TMEM
!
!       VARIABLES AND ARRAYS THAT WILL BE LOADED WITH DATA.
!
      REAL,DIMENSION(:,:), ALLOCATABLE :: FCST
      REAL,DIMENSION(:), ALLOCATABLE :: OBS
      INTEGER,DIMENSION(:), ALLOCATABLE :: DATE
      REAL,DIMENSION(:,:), ALLOCATABLE :: TRAIN_FCST
      REAL,DIMENSION(:), ALLOCATABLE   :: TRAIN_OBS
      REAL,DIMENSION(:), ALLOCATABLE     :: WEIGHTS
      REAL,DIMENSION(:), ALLOCATABLE     :: SIGMA
      INTEGER,DIMENSION(:), ALLOCATABLE  :: IFLAGS
      INTEGER,DIMENSION(:), ALLOCATABLE   :: IGROUP
      INTEGER                  :: IER, I, J, N, IOFFSET, JJ0, JJ1
      INTEGER                  :: IEND, NTOT
      REAL :: XMISS, XCONV, XSIGMA, Y_EXPECT
      REAL,DIMENSION(NCDF) :: CDF_POINTS, XVALUES

!       SET THE MISSING VALUE
      XMISS=9999.

!       SET THE CONVERAGE CRITERIA FOR THE EM ALGORITHM
      XCONV=0.001

!       SET THE MAXIMUM NUMBER OF ITERATIONS ALLOWED
      NTOT=100

!      GET COMMAND LINE ARG
      CALL GETARG(1, FCST_FILE)
      CALL GETARG(2, OBS_FILE)
      CALL GETARG(3, NTRAIN_STR)
      CALL GETARG(4, NLINES_STR)
      CALL GETARG(5, NMEM_STR1)
      CALL GETARG(6, NMEM_STR2)
      READ(NTRAIN_STR, '(I3)') NTRAIN
      READ(NLINES_STR, '(I3)') NLINES
      READ(NMEM_STR1, '(I2)') NMEM1
      READ(NMEM_STR2, '(I2)') NMEM2
      NMEM = NMEM1 + NMEM2
      ALLOCATE (FCST(NLINES, NMEM))
      ALLOCATE (TRAIN_FCST(NTRAIN,NMEM))
      ALLOCATE (OBS(NTRAIN))
      ALLOCATE (TRAIN_OBS(NTRAIN))
      ALLOCATE(TMEM(NMEM))
      ALLOCATE(IGROUP(NMEM))
      ALLOCATE(WEIGHTS(NMEM))
      ALLOCATE(IFLAGS(NMEM))
      ALLOCATE(SIGMA(NMEM))
      ALLOCATE(DATE(NLINES))


!       SET THE GROUP MEMBERSHIP INFORMATION
      IGROUP(1:NMEM1) = 1
      IGROUP(NMEM1+1:NMEM1+NMEM2) = 2


!       SET THE CDF POINTS DESIRED
      CDF_POINTS=(/0.1, 0.25, 0.5, 0.75, 0.95/)
!
      !       OPEN THE ASCII DATA FILE
      OPEN(1, FILE = FCST_FILE, FORM = 'FORMATTED',&
           STATUS = 'OLD', IOSTAT = IER)
      OPEN(2, FILE = OBS_FILE, FORM='FORMATTED',&
           STATUS = 'OLD', IOSTAT = IER)


!       READ IN THE DATA
      DO I = 1, NLINES
        READ(1, *) TIME_LABEL, TMEM(1:NMEM)
        DATE(I) = TIME_LABEL
!        FCST(I, 1:NMEM) = SQRT(TMEM(1:NMEM))
        FCST(I, 1:NMEM) = TMEM(1:NMEM)
      ENDDO

      DO I = 1, NTRAIN
        READ(2, *) TDATE, TOBS
!        OBS(I) = SQRT(TOBS)
        OBS(I) = TOBS
      END DO
!
!
!       THE ENSEMBLE FORECASTS AND VERIFYING OBSERVATIONS ARE
!       NOW LOADED INTO THE ARRAYS FCST AND OBS.
!
!       FOR EACH LINE I FCST(I,1:NMEM) CONTAINS THE NMEM FORECASTS
!       WHILE OBS(I) CONTAINS THE VERIFYING OBSERVATION.
!
!
!       CACULATE IOFFSET. THIS IS USED TO INDEX FCST AND OBS.
!       WE ONLY WANT TO USE DATA FOR TRAINING THAT COULD HAVE
!       BEEN VERIFIED BY THE TIME THE NEW FORECAST IS ISSUED.
!
      IOFFSET=CEILING(PROJECTION/24.)

!       LOOP THROUGH DAYS IN THE SAMPLE
!      DO I=NTRAIN+IOFFSET,NLINES
      DO I=NTRAIN+1, NLINES
!
!
!         THESE ARE USED TO EXTRACT DATA FROM FCST AND OBS
!         FOR TRAINING.  FOR EACH ITERATION OF THIS LOOP:
!             I = THE DAY THAT FORECASTS ARE DESIRED FOR.
!           JJ0 = THE FIRST DAY TO USE FOR TRAINING.
!           JJ1 = THE LAST DAY TO USE FOR TRAINING.
!         WE WANT TO USE DATA THAT COULD HAVE BEEN VERIFIED
!         BEFORE WE MAKE THE NEXT FORECAST.
!
        JJ0 = I - NTRAIN - IOFFSET + 1
        JJ1 = I - IOFFSET
!
!         LOAD TRAIN_FCST AND TRAIN_OBS WITH THE TRAINING DATA.
!
!        TRAIN_FCST(1:NTRAIN, 1:NMEM) = FCST(JJ0:JJ1, 1:NMEM)
!        TRAIN_OBS(1:NTRAIN) = OBS(JJ0:JJ1)
        TRAIN_FCST(1:NTRAIN, 1:NMEM) = FCST(1:NTRAIN, 1:NMEM)
        TRAIN_OBS(1:NTRAIN) = OBS(1:NTRAIN)
!
!       FIT THE BMA WEIGHTS.  THE EM ALGORITHM WILL
!       ITERATE UNTIL THE CONVERGEN CRITERIA XCONV
!       IS MET OR UNTIL THE NUMBER OF ITERATIONS
!       EXCEEDS NTOT.  THE VALUE OF OF IEND WILL
!       BE 0 OR 1.
!         IEND = 0 :STOPPED DUE TO CONVERGENE CRITERIA
!         IEND = 1 :STOPPED DUE TO EXCEEDING MAXIMUM
!                   NUMBER OF ITERATIONS.
!
      CALL FIT_BMA_NORMAL(TRAIN_FCST, TRAIN_OBS, &
                          WEIGHTS, IGROUP, NTRAIN, &
                          NMEM, XMISS, XCONV, NTOT, &
                          XSIGMA, IEND, IER)
!
!       CHECK FOR ERRORS FROM BMA
      IF(IER.EQ.0)THEN
        CALL BMA_FORECAST_CDF(CDF_POINTS, FCST(I, 1:NMEM),&
                  XSIGMA, WEIGHTS, XVALUES, XMISS, &
                  NMEM, NCDF, IER)
!
        Y_EXPECT = 0
        DO J = 1, NMEM
!              Y_EXPECT = Y_EXPECT + WEIGHTS(J) * FCST(I, J) ** 2
            Y_EXPECT = Y_EXPECT + WEIGHTS(J) * FCST(I, J)
        ENDDO

        WRITE(*, 200) DATE(I), Y_EXPECT, XVALUES(1), XVALUES(2), &
                     XVALUES(3), XVALUES(4), XVALUES(5)
!
 200    FORMAT(X, I10, X, F10.2, X, 5(X,F10.2))
!
      ENDIF
!
      ENDDO
      CLOSE(1)
      CLOSE(2)
!      WRITE(*, *) WEIGHTS(:)
      END PROGRAM