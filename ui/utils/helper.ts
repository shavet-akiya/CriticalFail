function formatSessionDate(sessionId: string): string {
    const year = sessionId.substring(0, 4);
    const month = sessionId.substring(4, 6);
    const day = sessionId.substring(6, 8);

    // Format it to a readable format (e.g., "20/10/2025")
    return `${day}/${month}/${year}`;
}

export { formatSessionDate }