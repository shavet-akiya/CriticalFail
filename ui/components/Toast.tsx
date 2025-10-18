type ToastProps = {
    type?: "success" | "error" | "info";
    message: string;
};

export default function Toast({ type = "info", message }: ToastProps) {
    let alertClass;
    let icon;

    switch (type) {
        case "error":
            alertClass = "alert alert-error";
            icon = (
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 shrink-0 stroke-current"
                    fill="none"
                    viewBox="0 0 24 24"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                </svg>
            );
            break;

        case "success":
            alertClass = "alert alert-success";
            icon = (
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 shrink-0 stroke-current"
                    fill="none"
                    viewBox="0 0 24 24"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                </svg>
            );
            break;

        default:
            alertClass = "alert alert-info";
            icon = (
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 shrink-0 stroke-current"
                    fill="none"
                    viewBox="0 0 24 24"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M13 16h-1v-4h-1m1-4h.01M12 18a9 9 0 110-18 9 9 0 010 18z"
                    />
                </svg>
            );
            break;
    }

    return (
        <div className="toast toast-end">
            <div role="alert" className={alertClass}>
                {icon}
                <span>{message}</span>
            </div>
        </div>
    );
}
