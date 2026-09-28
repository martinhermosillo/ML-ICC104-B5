## Simple regression: Diabetes BMI

The scatter plot of BMI versus disease progression demonstrates a moderate positive linear correlation. While higher BMI values generally correspond to higher disease progression values, the relationship is not perfectly linear due to high variance and substantial vertical spread across all BMI levels. The positive sign of the simple OLS slope ($\hat{\beta}_1 \approx 949.44$) indicates that a one-unit increase in standardized BMI is associated with an expected increase of approximately 949.44 units in quantitative disease progression.

The `np.allclose` assertions establish that our hand-calculated analytical OLS formulas produce results identical to scikit-learn's `LinearRegression` solver within machine precision for both parameter estimates and predicted values. The calculated $R^2$ score of approximately $0.3439$ shows that standardized BMI explains roughly $34.39\%$ of the total variance in disease progression within this sample. However, $R^2$ does not establish out-of-sample predictive power, causal relationships, or that a linear model is the true underlying data generator.

## California Housing predictor selection

The three predictors selected for the multiple linear regression model are `MedInc` (median income), `HouseAge` (median house age), and `AveRooms` (average number of rooms).

In the feature scatter plots, `MedInc` displays a strong, positive linear trend against median house value, making it the single strongest individual predictor. `HouseAge` exhibits a subtle upward distribution across lower age brackets, capturing structural maturity dynamics. `AveRooms` shows a concentrated cluster around standard room counts (3–6 rooms) with positive slope trends, representing physical property size.

These three predictors work well together because they cover complementary structural, economic, and physical dimensions of house value—income level, structural age, and physical scale—without excessive redundancy. However, choosing predictors solely from single-variable scatter plots has a major limitation: it ignores potential multicollinearity between predictors and misses multi-dimensional feature interactions that are only visible when analyzing variables jointly.

## Multiple-regression comparison and diagnostics

Adding a column of ones to the design matrix $\mathbf{X}_b$ introduces an intercept parameter ($\beta_0$), allowing the fitted hyperplane to shift away from the origin and capture baseline target values when all predictors are zero. Solving the normal equations via `np.linalg.solve` uses numerical matrix decompositions rather than explicitly computing $(\mathbf{X}_b^T \mathbf{X}_b)^{-1}$, which drastically improves numerical stability, speeds up execution, and prevents loss of precision caused by direct matrix inversion. The two `np.allclose` checks prove complete numerical equivalence between our custom matrix solver and scikit-learn's fit across model coefficients and target predictions.

Perfect predictions would form a thin, straight line along the identity axis ($y = \hat{y}$). Our plot shows a general positive linear alignment around the identity line, but exhibits wide vertical dispersion and a strict horizontal ceiling at $y = 5.0$ caused by target clipping in the original dataset. The residual plot reflects this clipping through a pronounced diagonal upper boundary line and shows increasing variance across middle-range predictions.

The observation with the largest leverage value is index 1914, with a leverage of approximately $0.1699$. High-leverage observations exert disproportionate influence on the regression fit, meaning small errors or outliers at these extreme feature locations can heavily tilt the estimated coefficient hyperplane.

## Conclusion and next step

Both models successfully capture baseline linear trends across their target domains, confirmed by exact agreement between hand/matrix implementations and scikit-learn fits ($R^2 \approx 0.3439$ for Diabetes BMI and $R^2 \approx 0.51$ for California Housing). A major limitation of this analysis is the strict linear assumption, which fails to capture non-linear relationships, feature interactions, and artificial dataset constraints such as target clipping at $y = 5.0$. An appropriate next step is implementing non-linear transformations, polynomial feature interactions, or regularized algorithms combined with cross-validation to better capture complex feature interactions without overfitting.